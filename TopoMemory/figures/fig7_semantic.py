"""Figure 7: semantic retention at 900 tokens (zero finetuning, fixed parameters).

Numbers come from experiments/exp_gpt2_semantic.py:
    vanilla GPT-2        0.7957
    GPT-2 + TopoMemory   0.8918   (+12.1% relative)
"""
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'out')
os.makedirs(save_dir, exist_ok=True)

MODELS = ['Vanilla GPT-2', 'GPT-2 + TopoMemory']
SIM = [0.7957, 0.8918]
COLORS = ['#7f8c8d', '#c0392b']

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
bars = ax.bar(MODELS, SIM, color=COLORS, width=0.55, edgecolor='#2c3e50', linewidth=1.2)

for bar, val in zip(bars, SIM):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.008,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

ax.plot([0.72, 1.28], [0.84, 0.84], color='#c0392b', linewidth=1.2)
ax.text(1, 0.85, '+12.1%', ha='center', va='bottom', fontsize=12,
        color='#c0392b', weight='bold')

ax.set_ylabel('Cosine Similarity (Instruction Retention)', fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_title('Figure 7: Semantic Retention at 900 Tokens\n(Zero Finetuning, Fixed Parameters)',
             fontsize=13, pad=15)
ax.grid(axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fig7_semantic_comparison.png'), bbox_inches='tight')
plt.close()

print('saved:', os.path.join(save_dir, 'fig7_semantic_comparison.png'))
