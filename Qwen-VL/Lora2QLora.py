import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, text, xy, size=(2.4,1), color="#CDE6F5", text_color="black", fontsize=9):
    """画带圆角的盒子"""
    x, y = xy
    w, h = size
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1, facecolor=color, edgecolor="black"
    )
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, color=text_color)

# 创建画布
fig, axs = plt.subplots(2, 1, figsize=(8,6))
plt.subplots_adjust(hspace=0.5)

################################
# 图 1：LoRA 在 Transformer 中
################################
axs[0].set_title("LoRA in Transformer Block", fontsize=14, weight="bold")

draw_box(axs[0], "Input", (-1, 0), size=(1.5,0.8), color="#FFF2CC")
draw_box(axs[0], "Multi-Head Attention\n(Wq, Wk, Wv, Wo)", (1,0), size=(3,1.2), color="#CCE5FF")
draw_box(axs[0], "Feed Forward\n(W1, W2)", (5,0), size=(2.5,1.2), color="#CCE5FF")
draw_box(axs[0], "Output", (8, 0), size=(1.5,0.8), color="#FFF2CC")

draw_box(axs[0], "LoRA ΔW\n(低秩 A,B)\n全精度", (1.5,-1.5), size=(2,0.9), color="#FFCCCC")
axs[0].annotate("", xy=(2,-0.15), xytext=(2,-1.5+0.9),
                arrowprops=dict(arrowstyle="->", lw=1.5))

axs[0].set_xlim(-2, 11)
axs[0].set_ylim(-3, 3)
axs[0].axis("off")

################################
# 图 2：Q-LoRA 在 Transformer 中
################################
axs[1].set_title("Q-LoRA in Transformer Block", fontsize=14, weight="bold")

draw_box(axs[1], "Input", (-1, 0), size=(1.5,0.8), color="#FFF2CC")
draw_box(axs[1], "Multi-Head Attention\n(Wq, Wk, Wv, Wo, 4-bit)", (1,0), size=(3,1.2), color="#CCFFCC")
draw_box(axs[1], "Feed Forward\n(W1, W2, 4-bit)", (5,0), size=(2.5,1.2), color="#CCFFCC")
draw_box(axs[1], "Output", (8, 0), size=(1.5,0.8), color="#FFF2CC")

draw_box(axs[1], "LoRA ΔW\n(低秩 A,B)\n全精度, 量化模型外加", (1.5,-1.5), size=(2.4,1), color="#FFCCCC")
axs[1].annotate("", xy=(2,-0.15), xytext=(2,-1.5+1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="red"))

axs[1].set_xlim(-2, 11)
axs[1].set_ylim(-3, 3)
axs[1].axis("off")

plt.suptitle("LoRA vs Q-LoRA 在 Transformer 模型中的位置", fontsize=16, weight="bold")

# 保存为当前路径的 PNG 文件（高分辨率）
plt.savefig("lora_q_lora.png", dpi=300, bbox_inches='tight')

print("✅ 图片已保存到当前路径：lora_q_lora.png")
