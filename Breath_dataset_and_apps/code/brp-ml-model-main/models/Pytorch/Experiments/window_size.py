import json
from statistics import mean
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

with open("block3.json", "r") as f:
    data = json.load(f)

window_sizes = sorted(map(int, data.keys()))
epsilon_accuracy = [
    mean(data[str(ws)]["max_epsilon_accuracy"])
    for ws in window_sizes
]

plt.figure(figsize=(6, 4))
plt.plot(window_sizes, epsilon_accuracy, marker='o', linewidth=2)

plt.xlabel("Window Size")
plt.ylabel("Epsilon Accuracy (%)")

tick_positions = [ws for ws in window_sizes if ws % 5 == 0]
plt.xticks(tick_positions)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("epsilon_accuracy_vs_window_size.pdf", dpi=300, bbox_inches="tight")
plt.savefig("epsilon_accuracy_vs_window_size.png", dpi=300, bbox_inches="tight")

print("test")