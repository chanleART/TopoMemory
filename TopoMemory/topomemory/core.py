"""Fixed-threshold TopoMemory layer (reference implementation).

Input / output shape: [Batch, Seq_len, Hidden_dim].
Mechanism: threshold-gated inscription + continuous decay + dual-threshold
hysteretic self-locking.
"""

import torch
import torch.nn as nn


class TopoMemoryLayer(nn.Module):
    def __init__(self, hidden_dim, theta_on=2.5, theta_off=0.3,
                 decay_rate=0.005, lock_decay=1e-5, kappa=1.0,
                 use_soft_gate=False, gate_temperature=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.theta_on = theta_on        # upper threshold: write and lock
        self.theta_off = theta_off      # lower threshold: unlock and leave
        self.decay_rate = decay_rate    # decay rate while unlocked
        self.lock_decay = lock_decay    # near-zero decay rate while locked
        self.kappa = kappa              # write strength
        self.use_soft_gate = use_soft_gate
        self.gate_temperature = gate_temperature

    def _compute_gate(self, A_t):
        """Soft gate during training, hard gate otherwise."""
        if self.use_soft_gate and self.training:
            # sigmoid approximates the step function, keeping the path differentiable
            return torch.sigmoid((A_t - self.theta_on) / self.gate_temperature)
        else:
            # hard gate at inference, numerically aligned with the pure-Python version
            return (A_t > self.theta_on).float()

    def forward(self, x):
        """
        Args:
            x: [B, T, D] batch x sequence length x hidden dim
        Returns:
            out: [B, T, D] hidden states with topological memory injected
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # per-sample memory state
        M = torch.zeros(B, D, device=device, dtype=dtype)          # memory field [B, D]
        locked = torch.zeros(B, 1, device=device, dtype=torch.bool)  # lock flag [B, 1]
        outputs = []

        for t in range(T):
            x_t = x[:, t, :]  # [B, D]

            # 1. topological strength A(t)
            A_t = torch.norm(x_t, dim=-1, keepdim=True)  # [B, 1]

            # 2. threshold gate decides whether to write
            write_mask = self._compute_gate(A_t)  # [B, 1]

            # 3. write into the memory field
            write_term = self.kappa * torch.tanh(x_t) * write_mask
            M = M + write_term

            # 4. update the hysteretic lock state
            locked = locked | (write_mask > 0.5)
            M_norm = torch.norm(M, dim=-1, keepdim=True)  # [B, 1]
            unlock_mask = (M_norm < self.theta_off)
            locked = locked & ~unlock_mask

            # 5. two-rate decay: near-zero while locked, normal while unlocked
            decay = torch.where(locked, 1.0 - self.lock_decay, 1.0 - self.decay_rate)
            M = M * decay

            # 6. residual injection of the memory field
            out_t = x_t + M
            outputs.append(out_t)

        return torch.stack(outputs, dim=1)  # [B, T, D]

    def reset_memory(self, batch_size=1, device=None):
        """Manually reset the memory state for streaming inference."""
        if device is None:
            device = next(self.parameters()).device
        self.M = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.locked = torch.zeros(batch_size, 1, device=device, dtype=torch.bool)
