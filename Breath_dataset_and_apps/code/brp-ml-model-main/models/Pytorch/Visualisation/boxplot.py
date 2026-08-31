import matplotlib.pyplot as plt
import numpy as np

# wyniki dla uczestników
results = {
    "Accuracy": [
        0.9789, 0.9177, 0.9603, 0.9700,
        0.9494, 0.9739, 0.9809, 0.9745
    ],
    "Transition Accuracy": [
        0.9316, 0.7760, 0.9113, 0.9021,
        0.9308, 0.9959, 1.0130, 0.9687
    ],
    "Cycle Accuracy": [
        0.8530, 0.7361, 0.9054, 0.8379,
        0.8135, 0.8642, 0.9571, 0.9546
    ]
}


# przygotowanie danych
data = [
    results["Accuracy"],
    results["Transition Accuracy"],
    results["Cycle Accuracy"]
]


labels = [
    "Accuracy",
    "Transition\nAccuracy",
    "Cycle\nAccuracy"
]


plt.figure(figsize=(8, 5))

plt.boxplot(
    data,
    labels=labels
)


plt.ylabel("Performance")
plt.title("LOSO Performance Across Subjects")

# procenty zamiast 0-1
plt.gca().set_yticklabels(
    [f"{x*100:.0f}" for x in plt.gca().get_yticks()]
)

plt.ylabel("Score (%)")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "loso_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

