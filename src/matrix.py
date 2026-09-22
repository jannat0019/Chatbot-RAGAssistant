import numpy as np
import matplotlib.pyplot as plt

# Confusion matrix from the best validation AUC epoch (42.5 -> Val AUC 0.9405)
# Format: [[TN, FP], [FN, TP]]
cm = np.array([[346, 56],
               [130, 756]])

class_names = ["Negative", "Positive"]  # rename to your actual class labels

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

# Ticks
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")

# Annotate cells with counts + percentage
total = cm.sum()
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = 100 * count / total
        ax.text(j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if count > thresh else "black",
                fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()