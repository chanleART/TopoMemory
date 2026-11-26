"""Experiment: component ablation (gate / hysteresis / decay).

Four variants share one parameter set and differ in exactly one component,
measured on the same strong-event retention task. A fifth figure reports the
memory-field norm, which is what motivates the decay branch at all.

Outputs: fig3_ablation_retention_v2.png, fig4_ablation_bar_v2.png,
         fig5_memory_norm_capacity.png
"""

import math
import random
import os
import matplotlib.pyplot as plt

# ============================================================
# Four ablation variants (single-variable principle)
# ============================================================
def prdbe_full(x_seq, A_c=2.5, decay_lambda=0.005, kappa=1.0,
               lock_decay=1e-5, theta_off=0.3):
    """Full model: threshold gate + hysteretic lock + continuous decay."""
    dim = len(x_seq[0])
    M = [0.0] * dim
    is_locked = False
    M_norm_history = []

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

        M_norm_history.append(math.sqrt(sum(v * v for v in M)))
    return M, M_norm_history


def prdbe_no_gate(x_seq, decay_lambda=0.005, kappa=1.0,
                  lock_decay=1e-5, theta_off=0.3):
    """Ablation 1: gate off, every token is written."""
    dim = len(x_seq[0])
    M = [0.0] * dim
    is_locked = False
    M_norm_history = []

    for t in range(len(x_seq)):
        x_t = x_seq[t]
        M = [M[d] + kappa * math.tanh(x_t[d]) for d in range(dim)]
        is_locked = True

        if is_locked:
            M = [m * (1.0 - lock_decay) for m in M]
            M_norm = math.sqrt(sum(v * v for v in M))
            if M_norm < theta_off:
                is_locked = False
        else:
            M = [m * (1.0 - decay_lambda) for m in M]

        M_norm_history.append(math.sqrt(sum(v * v for v in M)))
    return M, M_norm_history


def prdbe_no_hysteresis(x_seq, A_c=2.5, decay_lambda=0.005, kappa=1.0):
    """Ablation 2: hysteresis off, one single decay rate throughout."""
    dim = len(x_seq[0])
    M = [0.0] * dim
    M_norm_history = []

    for t in range(len(x_seq)):
        x_t = x_seq[t]
        A_t = math.sqrt(sum(v * v for v in x_t))

        if A_t > A_c:
            M = [M[d] + kappa * math.tanh(x_t[d]) for d in range(dim)]

        M = [m * (1.0 - decay_lambda) for m in M]
        M_norm_history.append(math.sqrt(sum(v * v for v in M)))
    return M, M_norm_history


def prdbe_no_forgetting(x_seq, A_c=2.5, kappa=1.0):
    """Ablation 3: decay off, nothing is ever forgotten."""
    dim = len(x_seq[0])
    M = [0.0] * dim
    M_norm_history = []

    for t in range(len(x_seq)):
        x_t = x_seq[t]
        A_t = math.sqrt(sum(v * v for v in x_t))

        if A_t > A_c:
            M = [M[d] + kappa * math.tanh(x_t[d]) for d in range(dim)]

        M_norm_history.append(math.sqrt(sum(v * v for v in M)))
    return M, M_norm_history


# ============================================================
# Helpers
# ============================================================
def cosine_similarity(a, b):
    dot = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = math.sqrt(sum(v * v for v in a))
    norm_b = math.sqrt(sum(v * v for v in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_sequence(seq_len, dim, strong_event_pos, seed=42):
    random.seed(seed)
    x_seq = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(seq_len)]
    strong_event = [random.gauss(0, 1) * 5.0 for _ in range(dim)]
    for d in range(dim):
        x_seq[strong_event_pos][d] = strong_event[d]
    return x_seq, strong_event


# ============================================================
# Ablation driver
# ============================================================
print("=" * 70)
print("PRDBE topological memory gate - component ablation (revised)")
print("=" * 70)

lengths = [2000, 5000, 10000, 20000]
dim = 32
strong_event_pos = 1000

groups = {
    "Full Model": (prdbe_full, "#c0392b", "o", "-"),
    "No Gate": (prdbe_no_gate, "#2980b9", "s", "-"),
    "No Hysteresis": (prdbe_no_hysteresis, "#7f8c8d", "^", "-"),
    "No Forgetting": (prdbe_no_forgetting, "#8e44ad", "D", "--"),
}

sim_results = {name: [] for name in groups}
norm_results = {name: [] for name in groups}

for seq_len in lengths:
    x_seq, strong_event = generate_sequence(seq_len, dim, strong_event_pos, seed=42)
    print(f"\nsequence length {seq_len:>5}:")
    for name, (func, _, _, _) in groups.items():
        M, norm_hist = func(x_seq)
        sim = cosine_similarity(M, strong_event)
        sim_results[name].append(sim)
        norm_results[name].append(norm_hist[-1])
        print(f"  {name:<18} similarity: {sim:.4f} | field norm: {norm_hist[-1]:.4f}")

# ============================================================
# Three figures
# ============================================================
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'out')
os.makedirs(save_dir, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.linewidth': 0.8
})

# --- Figure 3: retention per ablation group ---
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
for name, (_, color, marker, ls) in groups.items():
    ms = 8 if name == "Full Model" else 6
    lw = 2.2 if name == "Full Model" else 1.8
    ax.plot(lengths, sim_results[name], marker=marker, linestyle=ls,
            color=color, linewidth=lw, markersize=ms, label=name)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Sequence Length', fontsize=13)
ax.set_ylabel('Cosine Similarity to Strong Event', fontsize=13)
ax.set_title('Ablation Study: Long-Range Event Retention', fontsize=14, pad=10)
ax.legend(frameon=False, fontsize=10)
ax.set_ylim(-0.1, 1.0)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fig3_ablation_retention_v2.png'), bbox_inches='tight')
plt.close()

# --- Figure 4: component contribution at 20000 steps ---
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
bar_names = list(groups.keys())
bar_values = [sim_results[name][-1] for name in bar_names]
bar_colors = [groups[name][1] for name in bar_names]

bars = ax.bar(bar_names, bar_values, color=bar_colors, width=0.6, edgecolor='white', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.015,
            f'{height:.3f}', ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Cosine Similarity (20000 steps)', fontsize=13)
ax.set_title('Ablation Study: Component Contribution at 20000 Steps', fontsize=14, pad=10)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fig4_ablation_bar_v2.png'), bbox_inches='tight')
plt.close()

# --- Figure 5: field norm, i.e. why a decay branch is needed at all ---
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(lengths, norm_results["Full Model"], 'o-', color='#c0392b',
        linewidth=2, markersize=7, label='Full Model (with decay)')
ax.plot(lengths, norm_results["No Forgetting"], 'D--', color='#8e44ad',
        linewidth=1.8, markersize=6, label='No Forgetting (zero decay)')

ax.set_xlabel('Sequence Length', fontsize=13)
ax.set_ylabel('Memory Field Norm (L2)', fontsize=13)
ax.set_title('Memory Capacity Stability: Norm Growth Over Sequence', fontsize=14, pad=10)
ax.legend(frameon=False, fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fig5_memory_norm_capacity.png'), bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print(f"Ablation figures written to: {save_dir}")
print("  1. fig3_ablation_retention_v2.png  - retention curves, styled to avoid overlap")
print("  2. fig4_ablation_bar_v2.png        - bar comparison at 20000 steps")
print("  3. fig5_memory_norm_capacity.png   - field norm, the case for the decay branch")
print("=" * 70)
