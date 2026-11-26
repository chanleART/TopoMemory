"""Single-run sanity check in pure Python (no torch required).

Builds a 2000-step noise stream with one strong event at step 1000 and reports
the cosine similarity between the resulting memory field and that event, after
1000 steps of noise have washed over it.

Outputs: console only.
"""

import time
import math
import random

# Pure-Python full attention (crude O(N^2) memory mixing)
def full_attention_sim(x_seq):
    # Simplified: only the correlation with the last step, easily swamped by noise
    seq_len = len(x_seq)
    dim = len(x_seq[0])
    out = [0.0] * dim
    # Use the last step as the representative query to save time
    last_x = x_seq[-1]
    for j in range(seq_len):
        score = sum(last_x[d] * x_seq[j][d] for d in range(dim)) / math.sqrt(dim)
        weight = math.exp(score)  # part of a simplified softmax
        for d in range(dim):
            out[d] += weight * x_seq[j][d]
    return out


# Pure-Python PRDBE topological memory gate, O(N)
def prdbe_memory_sim(x_seq, A_c=2.5, decay_lambda=0.05, kappa=0.8):
    dim = len(x_seq[0])
    M = [0.0] * dim
    for t in range(len(x_seq)):
        x_t = x_seq[t]
        # 1. topological strength A(t), L2 norm
        A_t = math.sqrt(sum(v * v for v in x_t))

        # 2. natural forgetting
        M = [m * (1.0 - decay_lambda) for m in M]

        # 3. threshold gate
        if A_t > A_c:
            # 4. inscription through tanh
            M = [M[d] + kappa * math.tanh(x_t[d]) for d in range(dim)]

    return M


# --- run ---
print("=== pure-Python quick run ===")
seq_len = 2000
dim = 32

# a stream of ordinary noise
random.seed(42)
x_seq = [[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(seq_len)]

# one very strong event at step 1000
strong_event_idx = 1000
for d in range(dim):
    x_seq[strong_event_idx][d] = random.gauss(0, 1) * 5.0

print(f"sequence length: {seq_len} steps")
print("one strong event injected, everything else is noise.")

start = time.time()
final_M = prdbe_memory_sim(x_seq, A_c=2.5, decay_lambda=0.005, kappa=1.0)
print(f"\nPRDBE gate finished in {time.time() - start:.4f} s")

# what did the field actually keep?
strong_event = x_seq[strong_event_idx]
norm_M = math.sqrt(sum(m * m for m in final_M))
norm_strong = math.sqrt(sum(s * s for s in strong_event))

if norm_M > 0 and norm_strong > 0:
    dot_product = sum(final_M[d] * strong_event[d] for d in range(dim))
    similarity = dot_product / (norm_M * norm_strong)
    print("\nafter 1000 steps of noise washing over it:")
    print(f"cosine similarity between the field and the strong event: {similarity:.4f}")
    print("(close to 1.0 means the hysteretic lock held the event in place.)")
