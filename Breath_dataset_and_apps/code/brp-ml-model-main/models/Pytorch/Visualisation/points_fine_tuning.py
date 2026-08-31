import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Data
# ============================================================

participants = [
    "P1", "P2", "P3", "P4",
    "P5", "P6", "P7", "P8"
]

baseline = np.array([
    82.0, 79.0, 91.0, 85.0,
    76.0, 88.0, 81.0, 90.0
])

server_side = np.array([
    88.0, 84.0, 93.0, 89.0,
    82.0, 91.0, 87.0, 94.0
])

on_device = np.array([
    86.0, 83.0, 94.0, 88.0,
    80.0, 90.0, 85.0, 92.0
])


# ============================================================
# Plot
# ============================================================

x = np.arange(3)

fig, ax = plt.subplots(figsize=(7, 4.5))

for i, participant in enumerate(participants):

    values = [
        baseline[i],
        server_side[i],
        on_device[i]
    ]

    ax.plot(
        x,
        values,
        marker="o",
        linewidth=1.5,
        markersize=5,
        label=participant
    )


# ============================================================
# Axes
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels([
    "Baseline",
    "Server-side",
    "On-device"
])

ax.set_ylabel("Accuracy (%)")

ax.set_xlabel("Model configuration")

ax.set_title(
    "Model performance before and after personalization"
)


# ============================================================
# Grid and limits
# ============================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.6
)

ax.set_ylim(
    max(0, min(baseline.min(), server_side.min(), on_device.min()) - 5),
    min(100, max(baseline.max(), server_side.max(), on_device.max()) + 3)
)


# ============================================================
# Legend
# ============================================================

ax.legend(
    title="Participant",
    ncol=2,
    frameon=False,
    loc="best"
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout()


# ============================================================
# Export
# ============================================================

fig.savefig(
    "personalization_per_participant.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "personalization_per_participant.svg",
    bbox_inches="tight"
)

plt.show()