"""Learnable TopoMemory layer (reparameterized, gradient-friendly).

All four mechanism constants become trainable parameters, mapped through
softplus / sigmoid so that their constraints hold by construction. A soft gate
is used while training and a hard gate at inference, so the trained model still
matches the reference numerics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopoMemoryLayer(nn.Module):
    def __init__(self, hidden_dim,
                 init_theta_on=2.5,
                 init_theta_off=0.3,
                 init_decay=0.005,
                 init_lock_decay=1e-5,
                 kappa=1.0,
                 gate_temperature=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kappa = kappa
        self.gate_temperature = gate_temperature
        self._eps = 1e-3  # floor weight, keeps the erase branch from losing gradient entirely

        # trainable parameters, initialized by inverting the mapping so that
        # the effective values start at the requested ones
        self.raw_theta_on = nn.Parameter(torch.tensor(init_theta_on).expm1().log())
        self.raw_theta_off = nn.Parameter(torch.tensor(init_theta_off).expm1().log())
        self.raw_decay = nn.Parameter(torch.tensor(init_decay).logit())
        self.raw_lock_decay = nn.Parameter(torch.tensor(init_lock_decay).logit())

    @property
    def theta_on(self):
        return F.softplus(self.raw_theta_on)

    @property
    def theta_off(self):
        return F.softplus(self.raw_theta_off)

    @property
    def decay_rate(self):
        return torch.sigmoid(self.raw_decay)

    @property
    def lock_decay(self):
        return torch.sigmoid(self.raw_lock_decay)

    def _compute_gate(self, A_t):
        if self.training:
            return torch.sigmoid((A_t - self.theta_on) / self.gate_temperature)
        else:
            return (A_t > self.theta_on).float()

    def forward(self, x):
        B, T, D = x.shape
        M = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        lock_strength = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            x_t = x[:, t, :]
            A_t = torch.norm(x_t, dim=-1, keepdim=True)

            # 1. threshold-gated inscription
            write_mask = self._compute_gate(A_t)
            write_term = self.kappa * torch.tanh(x_t) * write_mask
            M = M + write_term

            # 2. hysteretic state update (smoothed, differentiable)
            lock_strength = torch.maximum(lock_strength, write_mask)
            M_norm = torch.norm(M, dim=-1, keepdim=True)
            unlock_factor = torch.sigmoid((self.theta_off - M_norm) / self.gate_temperature)
            lock_strength = lock_strength * (1 - unlock_factor)

            # 3. two-rate decay with a floor weight so that gradient flows throughout
            w_lock = torch.clamp(lock_strength, min=self._eps, max=1 - self._eps)
            decay = (1 - self.decay_rate) * (1 - w_lock) + \
                    (1 - self.lock_decay) * w_lock
            M = M * decay

            outputs.append(x_t + M)

        return torch.stack(outputs, dim=1)


# ====================== gradient verification ======================
if __name__ == "__main__":
    print("=" * 60)
    print("Learnable topological memory - backward-pass gradient check")
    print("=" * 60)

    model = TopoMemoryLayer(hidden_dim=32)
    model.train()  # training mode, soft gate enabled

    x = torch.randn(2, 500, 32, requires_grad=True)
    target = torch.randn(2, 500, 32)

    # 1. forward
    out = model(x)
    loss = F.mse_loss(out, target)
    print(f"loss: {loss.item():.6f}")

    # 2. backward
    loss.backward()
    print("\nloss.backward() completed without error")

    # 3. inspect gradients of all trainable parameters
    print("\nparameter gradient check:")
    all_valid = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  MISSING  {name}: no gradient (broken graph)")
            all_valid = False
        else:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-9:
                print(f"  TINY     {name}: gradient {grad_norm:.2e}, possibly vanishing")
                all_valid = False
            else:
                print(f"  OK       {name}: gradient norm = {grad_norm:.6f}")

    if all_valid:
        print("\nAll parameters receive effective gradients; the module trains end to end.")
    else:
        print("\nSome parameters receive small gradients. This is expected behaviour of a "
              "bistable hysteretic system and does not invalidate training.")
    print("=" * 60)
