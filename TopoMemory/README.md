# TopoMemory

Hysteretic self-locking memory for long-context sequence modeling — reference
implementation accompanying the preprint.

**DOI:** [10.5281/zenodo.24529583](https://doi.org/10.5281/zenodo.24529583)

> **Status.** This repository is the reference code for a mechanism paper. It is
> not a production-ready module. What it supports is a specific claim: that a
> threshold-gated, path-dependent memory field with a dual-threshold hysteresis
> lock retains sparse critical events across long noise streams, and that the
> rule keeps working when inserted into a real language model at fixed
> parameters. It does not claim gains on language-modeling benchmarks.

## Mechanism

Instead of treating memory as a flat key-value cache, TopoMemory reformulates it
as a state evolution process with three parts:

1. **Threshold-gated inscription.** A vector is written into the memory field
   only when its norm crosses the write threshold `A_c`. Sub-threshold noise
   never enters the field, so averaging over irrelevant tokens cannot dilute
   what matters.
2. **A path-dependent continuous field.** Each write is `kappa * tanh(x_t)`
   accumulated into `M`. The field therefore encodes event order and
   co-occurrence, not just content.
3. **Dual-threshold hysteresis.** Once a write occurs, the field enters a locked
   state whose decay rate is about two orders of magnitude below the unlocked
   rate. It leaves that state only when the field norm falls below the unlock
   threshold `theta_off`. Writing is easy, erasing is hard — so a single strong
   event survives a long stretch of interference, while the field stays bounded
   instead of growing without limit.

The result is linear time in sequence length, and a decay curve with two regimes
rather than one exponential.

## Repository layout

```
topomemory/
├── core.py        fixed-threshold reference module (hard gate at inference)
└── learnable.py   reparameterized variant (soft gate in training, gradient check)
experiments/
├── exp_numeric_alignment.py    torch vs pure-Python numerical agreement
├── exp_retention.py            length sweep, retention curves, linear scaling
├── exp_ablation.py             component ablation (gate / hysteresis / decay)
├── exp_timing_vectorized.py    loop vs vectorized vs full attention timing
├── exp_pure_python.py          single-run pure-Python sanity check
├── exp_convergence.py          end-to-end training on a two-event task
└── exp_gpt2_semantic.py        zero-finetuning insertion into GPT-2
figures/
├── fig6_paradigm.py            paradigm diagram
└── fig7_semantic.py            semantic retention bar chart
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x. The GPT-2 experiment additionally needs
`transformers` and downloads the 124M checkpoint on first run.

## Quick start

```python
import torch
from topomemory import TopoMemoryLayer

layer = TopoMemoryLayer(hidden_dim=32)          # A_c=2.5, theta_off=0.3
x = torch.randn(2, 2000, 32)                   # [batch, seq_len, dim]
y = layer(x)                                   # same shape
```

The learnable variant takes the same arguments and exposes `theta_on`,
`theta_off`, `decay_rate` and `lock_decay` as reparameterized properties:

```python
from topomemory.learnable import TopoMemoryLayer as LearnableTopoMemory
```

## Reproducing the reported results

All scripts write figures to `figures/out/` and print their numbers to stdout.
Run each from the repository root.

| Script | Produces | Reported as |
|---|---|---|
| `experiments/exp_numeric_alignment.py` | torch / pure-Python deviation | numerical alignment |
| `experiments/exp_retention.py` | `fig1_complexity_scaling_v2.png`, `fig2_event_retention_v2.png` | retention vs length, linear scaling |
| `experiments/exp_ablation.py` | `fig3_ablation_retention_v2.png`, `fig4_ablation_bar_v2.png`, `fig5_memory_norm_capacity.png` | component ablation |
| `experiments/exp_timing_vectorized.py` | `fig8_vectorized_speedup.png` | three-implementation timing |
| `experiments/exp_convergence.py` | `fig6_convergence_comparison.png` | end-to-end training convergence |
| `experiments/exp_pure_python.py` | single-run retention value | sanity check |
| `experiments/exp_gpt2_semantic.py` | `0.7957 → 0.8918` (+12.1%) | zero-finetuning semantic check |
| `figures/fig6_paradigm.py`, `figures/fig7_semantic.py` | `fig6_paradigm_final.png`, `fig7_semantic_comparison.png` | paradigm and bar chart |

Runs are deterministic where a seed is set (`seed=42`). The GPT-2 check fixes
its seed as well, but it remains a single instruction, single seed, single
forward pass.

## Citation

```bibtex
@misc{zhang2026topomemory,
  title        = {TopoMemory: Hysteretic Self-Locking for Long-Context Sequence Modeling},
  author       = {Zhang, Lei},
  year         = {2026},
  howpublished = {Zenodo preprint},
  doi          = {10.5281/zenodo.24529583},
  url          = {https://doi.org/10.5281/zenodo.24529583}
}
```

## Scope and known gaps

- The mechanism-level experiments are vector-level proxy validation on synthetic
  sequences, not a language-modeling benchmark. The GPT-2 check is a single
  zero-finetuning insertion added after the first release.
- The script producing the step-wise implementation timing curve is not part of
  this release.
- A parameter sweep has not been run; the reported numbers use one parameter
  set.
- `exp_gpt2_semantic.py` sets `HF_ENDPOINT` to a mirror for restricted networks;
  delete that line if you reach Hugging Face directly.

## License

MIT — see [LICENSE](LICENSE).
