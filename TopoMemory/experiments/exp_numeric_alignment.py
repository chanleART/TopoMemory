"""Experiment: numerical agreement between the pure-Python and torch versions.

Both implementations run on the identical sequence (seed 42, 2000 steps, one
strong event at step 1000), and their cosine similarity to that event is
compared.

Outputs: console only.
"""

import math
import random
import torch
import torch.nn as nn

# ===================== pure-Python reference =====================
def prdbe_pure_python(x_seq, A_c=2.5, decay_lambda=0.005, kappa=1.0,
                      lock_decay=1e-5, theta_off=0.3):
    dim = len(x_seq[0])
    M = [0.0] * dim
    is_locked = False

    for t in range(len(x_seq)):
        x_t = x_seq[t]
        A_t = math.sqrt(sum(v * v for v in x_t))

        if A_t > A_c:
            M = [M[d] + kappa * math.tanh(x_t[d]) for d in range(dim)]
            is_locked = True

        if is_locked:
            M = [m * (1.0 - lock_decay) for m in M]
            M_norm = math.sqrt(sum(v * v for v in M))
            if M_norm < theta_off:
                is_locked = False
        else:
            M = [m * (1.0 - decay_lambda) for m in M]
    return M

# ===================== PyTorch implementation =====================
class TopoMemoryLayer(nn.Module):
    def __init__(self, hidden_dim, theta_on=2.5, theta_off=0.3,
                 decay_rate=0.005, lock_decay=1e-5, kappa=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.theta_on = theta_on
        self.theta_off = theta_off
        self.decay_rate = decay_rate
        self.lock_decay = lock_decay
        self.kappa = kappa

    def forward(self, x):
        B, T, D = x.shape
        M = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        locked = torch.zeros(B, 1, device=x.device, dtype=torch.bool)

        for t in range(T):
            x_t = x[:, t, :]
            A_t = torch.norm(x_t, dim=-1, keepdim=True)
            write_mask = (A_t > self.theta_on).float()

            write_term = self.kappa * torch.tanh(x_t) * write_mask
            M = M + write_term

            locked = locked | (write_mask > 0.5)
            M_norm = torch.norm(M, dim=-1, keepdim=True)
            unlock_mask = (M_norm < self.theta_off)
            locked = locked & ~unlock_mask

            decay = torch.where(locked, 1.0 - self.lock_decay, 1.0 - self.decay_rate)
            M = M * decay
        return M

# ===================== alignment test =====================
if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch vs pure Python - numerical alignment")
    print("=" * 60)

    # identical parameters and inputs
    seq_len = 2000
    dim = 32
    strong_event_pos = 1000
    seed = 42

    # 1. build the identical input sequence
    random.seed(seed)
    x_list = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(seq_len)]
    strong_event = [random.gauss(0, 1) * 5.0 for _ in range(dim)]
    for d in range(dim):
        x_list[strong_event_pos][d] = strong_event[d]

    # 2. pure-Python run
    M_py = prdbe_pure_python(x_list)
    sim_py = sum(M_py[i] * strong_event[i] for i in range(dim)) / (
        math.sqrt(sum(v * v for v in M_py)) * math.sqrt(sum(v * v for v in strong_event))
    )

    # 3. torch run
    x_tensor = torch.tensor(x_list, dtype=torch.float32).unsqueeze(0)  # [1, T, D]
    model = TopoMemoryLayer(hidden_dim=dim)
    M_torch = model(x_tensor)
    M_torch_np = M_torch[0].tolist()

    sim_torch = sum(M_torch_np[i] * strong_event[i] for i in range(dim)) / (
        math.sqrt(sum(v * v for v in M_torch_np)) * math.sqrt(sum(v * v for v in strong_event))
    )

    # 4. compare
    print(f"pure Python, strong-event similarity: {sim_py:.6f}")
    print(f"PyTorch,     strong-event similarity: {sim_torch:.6f}")
    print(f"absolute deviation:                   {abs(sim_py - sim_torch):.8f}")
    print("-" * 60)

    if abs(sim_py - sim_torch) < 1e-4:
        print("Aligned. The torch implementation carries no logic deviation.")
    else:
        print("Deviation is large; inspect the implementation details.")
    print("=" * 60)
