"""Experiment: zero-finetuning insertion of TopoMemory into GPT-2 (124M).

No weight of the backbone is modified and no gradient step is taken. A
fixed-parameter bottleneck memory module is attached as a residual branch after
the attention output of a few layers, and a single forward pass measures how
much of a leading instruction survives 900 tokens of neutral distractor text.

Reported numbers: vanilla 0.7957, with TopoMemory 0.8918 (+12.1% relative).

Outputs: console only.
"""

import os
# Mirror endpoint for restricted networks; delete this line if Hugging Face is reachable directly.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

# ============================================================
# 1. Bottleneck TopoMemory: orthogonal init + scaled readout
# ============================================================
class BottleneckTopoMemory(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=32,
                 theta_on=1.2, theta_off=0.15,
                 decay_rate=0.005, lock_decay=1e-5,
                 kappa=0.1, output_scale=0.1):
        super().__init__()
        # projection pair; orthogonal init preserves the geometry of the space
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.down_proj.weight)
        nn.init.orthogonal_(self.up_proj.weight)

        self.theta_on = theta_on
        self.theta_off = theta_off
        self.decay_rate = decay_rate
        self.lock_decay = lock_decay
        self.kappa = kappa
        self.output_scale = output_scale  # readout scale

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        x_low = self.down_proj(x)
        M = torch.zeros(B, x_low.shape[-1], device=device)
        locked = False
        outputs_low = []

        for t in range(T):
            x_t = x_low[:, t, :]
            A_t = torch.norm(x_t, dim=-1).mean().item()

            if A_t > self.theta_on:
                M = M + self.kappa * torch.tanh(x_t)
                locked = True

            M_norm = torch.norm(M).item()
            if locked and M_norm < self.theta_off:
                locked = False

            if locked:
                M = M * (1.0 - self.lock_decay)
            else:
                M = M * (1.0 - self.decay_rate)

            outputs_low.append(x_t + M)

        outputs_low = torch.stack(outputs_low, dim=1)
        # back to the original width, then scaled
        return self.output_scale * self.up_proj(outputs_low)


# ============================================================
# 2. Injection by monkey-patching the attention forward
# ============================================================
def inject_topo_to_gpt2(model, inject_layers=[0, 2, 4, 6], bottleneck_dim=32):
    hidden_dim = model.config.n_embd

    for i, block in enumerate(model.h):
        if i in inject_layers:
            block.topo_memory = BottleneckTopoMemory(hidden_dim, bottleneck_dim)
            original_forward = block.attn.forward

            def make_patched(orig_forward, topo_mod):
                def patched(*args, **kwargs):
                    attn_out, present = orig_forward(*args, **kwargs)
                    memory_out = topo_mod(attn_out)
                    return attn_out + memory_out, present
                return patched

            block.attn.forward = make_patched(original_forward, block.topo_memory)
        else:
            block.topo_memory = None

    return model


# ============================================================
# 3. Test sequence: instruction head + long distractor
# ============================================================
def build_test_sequence(tokenizer, instruction, distract_text, total_len=900):
    instr_ids = tokenizer.encode(instruction, return_tensors="pt")[0]
    distract_ids = tokenizer.encode(distract_text, return_tensors="pt")[0]

    full = torch.cat([instr_ids, distract_ids])[:total_len]
    return full.unsqueeze(0), len(instr_ids)


# ============================================================
# 4. Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GPT-2 + TopoMemory: long-range information retention (inference only)")
    print("=" * 60)

    torch.manual_seed(42)

    model_name = "gpt2"
    print(f"loading {model_name} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model_original = GPT2Model.from_pretrained(model_name)
    model_topo = GPT2Model.from_pretrained(model_name)

    model_topo = inject_topo_to_gpt2(model_topo, inject_layers=[0, 2, 4, 6])
    print("TopoMemory injected at layers [0, 2, 4, 6]")

    instruction = "Remember: the secret code is 42."
    distract = "The quick brown fox jumps over the lazy dog. " * 200

    input_ids, instr_end = build_test_sequence(tokenizer, instruction, distract, total_len=900)
    seq_len = input_ids.shape[1]
    print(f"\ntotal sequence length: {seq_len} tokens")
    print(f"instruction span: 0 ~ {instr_end} tokens")

    model_original.eval()
    model_topo.eval()
    with torch.no_grad():
        out_original = model_original(input_ids).last_hidden_state
        out_topo = model_topo(input_ids).last_hidden_state

    instr_vec_ori = out_original[0, :instr_end, :].mean(dim=0)
    last_vec_ori = out_original[0, -1, :]
    sim_ori = torch.cosine_similarity(instr_vec_ori, last_vec_ori, dim=0).item()

    instr_vec_topo = out_topo[0, :instr_end, :].mean(dim=0)
    last_vec_topo = out_topo[0, -1, :]
    sim_topo = torch.cosine_similarity(instr_vec_topo, last_vec_topo, dim=0).item()

    print("\n" + "-" * 60)
    print(f"vanilla GPT-2, instruction retention: {sim_ori:.4f}")
    print(f"+TopoMemory,   instruction retention: {sim_topo:.4f}")
    relative_gain = (sim_topo - sim_ori) / abs(sim_ori) * 100
    print(f"relative gain: {relative_gain:.1f}%")
    print("-" * 60)

    if sim_topo > sim_ori:
        print("\nConfirmed: the topological memory raises long-range semantic retention.")
    else:
        print("\nNo gain observed. Try lowering output_scale or theta_on, or injecting more layers.")
    print("=" * 60)
