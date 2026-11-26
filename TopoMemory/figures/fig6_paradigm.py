"""Figure 6: the long-range instruction retention paradigm.

Schematic only: a critical instruction at the head of the sequence, a long
neutral distractor, and a detection point at the end where cosine similarity
is measured.

Outputs: fig6_paradigm_final.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'out')
os.makedirs(save_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 4.2), dpi=300)

seq_total = 900
# the instruction block is drawn wider than its true token share so the label fits;
# the real token count is printed inside the block
instr_visual_width = 70
instr_real_tokens = 8

# 1. two blocks
# critical instruction
rect_instr = patches.Rectangle(
    (0, 0.5), instr_visual_width, 0.5,
    linewidth=1.5, edgecolor='#c0392b', facecolor='#e74c3c', alpha=0.9
)
ax.add_patch(rect_instr)

# distractor text
rect_distract = patches.Rectangle(
    (instr_visual_width, 0.5), seq_total - instr_visual_width, 0.5,
    linewidth=1.5, edgecolor='#7f8c8d', facecolor='#dfe4e7', alpha=0.8
)
ax.add_patch(rect_distract)

# 2. labels
ax.text(
    instr_visual_width / 2, 0.75, f'Critical\nInstruction\n({instr_real_tokens} tokens)',
    ha='center', va='center', fontsize=8.5, color='white', weight='bold', linespacing=1.2
)
ax.text(
    (instr_visual_width + seq_total) / 2, 0.75,
    'Neutral Distractor Text (~892 tokens)',
    ha='center', va='center', fontsize=11, color='#2c3e50'
)

# 3. detection point
detect_x = seq_total
ax.plot(detect_x, 0.75, 'v', color='#2980b9', markersize=13)
ax.text(
    detect_x - 10, 0.38, 'Detection Point',
    ha='right', va='center', fontsize=11, color='#2980b9'
)

# 4. similarity measurement arrow
ax.annotate(
    '', xy=(detect_x, 0.52), xytext=(0, 0.52),
    arrowprops=dict(arrowstyle='<->', color='#2c3e50', linewidth=1.2, linestyle='--')
)
ax.text(
    seq_total / 2, 0.28,
    'Cosine Similarity (measures long-range information retention)',
    ha='center', va='center', fontsize=10.5, color='#2c3e50'
)

# 5. axes: full 0-900 range with margin on both sides
ax.set_xlim(-30, seq_total + 30)
ax.set_ylim(0, 1.1)
ax.set_yticks([])
ax.set_xticks([0, 200, 400, 600, 800, 900])
ax.set_xlabel('Sequence Position (tokens)', fontsize=13, labelpad=10)
ax.set_title(
    'Figure 6: Long-Range Instruction Retention Paradigm',
    fontsize=14, pad=15
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'fig6_paradigm_final.png'), bbox_inches='tight')
plt.close()

print(f"saved: {os.path.join(save_dir, 'fig6_paradigm_final.png')}")
