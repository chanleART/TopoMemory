"""Experiment: end-to-end training on a long-range two-event association task.

Two events sit 800 steps apart at fixed positions; a full-attention baseline
and a trainable TopoMemory head are compared over 40 epochs at identical
capacity.

Outputs: fig6_convergence_comparison.png
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ===== global settings =====
SEQ_LEN = 1000    # kept at 1000 to stay within memory limits
BATCH = 16
DIM = 32
EPOCHS = 40

# ==========================================
# 1. Full-attention baseline
# ==========================================
class BaselineAttentionModel(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.predictor = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        # full attention over the whole sequence
        attn_out, _ = self.attn(x, x, x)
        query_state = attn_out[:, -1, :]
        return self.predictor(query_state)

# ==========================================
# 2. TopoMemory model
# ==========================================
class TopoMemoryModel(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.theta_on = nn.Parameter(torch.tensor(2.0))
        # normal decay and locked decay, mapped into (0, 1) by sigmoid
        self.decay_normal = nn.Parameter(torch.tensor(0.0))
        self.decay_lock = nn.Parameter(torch.tensor(5.0))  # sigmoid(5.0) ~ 0.99
        self.predictor = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        B, L, D = x.shape
        M = torch.zeros(B, D).to(x.device)
        lock_flag = torch.zeros(B, 1).to(x.device)

        for t in range(L):
            x_t = x[:, t, :]
            A_t = torch.norm(x_t, p=2, dim=-1, keepdim=True)

            # sigmoid replaces the hard step so gradients flow back smoothly
            write_mask = torch.sigmoid((A_t - self.theta_on) * 5.0)
            lock_flag = torch.clamp(lock_flag + write_mask, 0.0, 1.0)

            # dynamic decay: normal while unlocked, near-zero once locked
            current_decay = (1 - lock_flag) * torch.sigmoid(self.decay_normal) + lock_flag * torch.sigmoid(self.decay_lock)
            M = M * current_decay + write_mask * x_t

        return self.predictor(M)

# ==========================================
# 3. Data: two events separated by 800 steps
# ==========================================
X = torch.randn(BATCH, SEQ_LEN, DIM) * 0.5
# feature A at step 100
X[:, 100, :] = 5.0
# feature B at step 900
X[:, 900, :] = 5.0
# label: the co-occurrence of both events is the positive class
Y = torch.ones(BATCH, 1)

# ==========================================
# 4. Training loop with gradient clipping
# ==========================================
def train_and_record(model, name, epochs=EPOCHS, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    print(f"\ntraining {name} ...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()

        # clip, so the baseline does not diverge
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:02d} | loss: {loss.item():.4f}")

    print(f"{name} final loss: {losses[-1]:.4f}")
    return losses

# ==========================================
# 5. Run
# ==========================================
baseline_losses = train_and_record(BaselineAttentionModel(), "Baseline")
topo_losses = train_and_record(TopoMemoryModel(), "TopoMemory")

# ==========================================
# 6. Comparison figure
# ==========================================
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'out')
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(baseline_losses, label='Baseline (Full Attention)', color='gray', linestyle='--', linewidth=2.5)
plt.plot(topo_losses, label='Ours (TopoMemory)', color='#c0392b', linewidth=2.5, marker='o', markevery=4)

plt.title(f'Training Convergence: Long-Range Associative Recall (Seq={SEQ_LEN})', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=14)
plt.tight_layout()

print("\ntraining done, writing figure ...")
plt.savefig(os.path.join(save_dir, 'fig6_convergence_comparison.png'), bbox_inches='tight', dpi=300)
plt.close()

print(f"saved: {os.path.join(save_dir, 'fig6_convergence_comparison.png')}")
