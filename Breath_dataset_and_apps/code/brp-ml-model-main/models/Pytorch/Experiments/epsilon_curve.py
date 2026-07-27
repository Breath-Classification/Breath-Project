import matplotlib.pyplot as plt
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error
def epsilon_accuracy(model_path, epsilon):
    """
    Calculates epsilon accuracy for a given epsilon.
    """
    y_pred, y_true, _, _ = load_model_and_predict(model_path)

    n_samples = len(y_pred)
    correct_prediction = 0

    for i in range(n_samples):
        if y_pred[i] == y_true[i]:
            correct_prediction += 1
        elif acceptable_error(y_pred, y_true, i, epsilon):
            correct_prediction += 1

    return 100 * correct_prediction / n_samples


def plot_epsilon_accuracy_curve(model_path, model_name="LSTM_BASE", max_epsilon=10):
    """
    Generates Epsilon Accuracy vs Epsilon Size plot.
    """

    epsilons = list(range(max_epsilon + 1))
    accuracies = []

    for eps in epsilons:
        acc = epsilon_accuracy(model_path, eps)
        accuracies.append(acc)

    # Parametry odpowiednie do publikacji
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        epsilons,
        accuracies,
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Epsilon Size")
    ax.set_ylabel("Epsilon Accuracy (%)")

    ax.set_xlim(0, max_epsilon)
    ax.set_xticks(epsilons)

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 5))

    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.savefig(
        f"{model_name}_epsilon_accuracy.pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        f"{model_name}_epsilon_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
plot_epsilon_accuracy_curve(model_path="LSTM_BASE_0.8831.pth")
