import json
from pathlib import Path
import numpy as np
from scipy.stats import t


# Must match the output directory selected in LOSO.py.
results_dir = Path("results/Transformer")


metrics_names = [
    "test_acc",
    "epsilon_accuracy",
    "transition_accuracy",
    "cycle_accuracy"
]


# tutaj będziemy trzymać wyniki uczestników
participant_results = {
    metric: []
    for metric in metrics_names
}


for fold in sorted(results_dir.iterdir()):

    if not fold.is_dir():
        continue


    # wyniki 5 treningów dla jednego uczestnika
    run_results = {
        metric: []
        for metric in metrics_names
    }


    for file in sorted(fold.glob("run_*_results.json")):

        with open(file) as f:
            data = json.load(f)

       # print(data)
        # indeks najlepszej epoki według epsilon accuracy
        best_epoch_idx = np.argmax(
            data["epsilon_accuracy"]
        )


        print(
            f"{file.name}: "
            f"best epoch = {data['epoch'][best_epoch_idx]}"
        )


        # pobieramy metryki z tej samej epoki
        for metric in metrics_names:

            value = data[metric][best_epoch_idx]

            run_results[metric].append(value)



    # średnia z 5 treningów dla jednego uczestnika
    for metric in metrics_names:

        participant_mean = np.mean(
            run_results[metric]
        )

        participant_results[metric].append(
            participant_mean
        )



# ==========================
# Statystyki LOSO
# ==========================

def calculate_statistics(values):

    values = np.array(values)

    mean = np.mean(values)

    sd = np.std(
        values,
        ddof=1
    )

    n = len(values)

    sem = sd / np.sqrt(n)

    t_value = t.ppf(
        0.975,
        df=n-1
    )

    margin = t_value * sem

    ci_low = mean - margin
    ci_high = mean + margin

    return mean, sd, ci_low, ci_high



# ==========================
# Wyniki końcowe
# ==========================

print("\n===== FINAL LOSO RESULTS =====")


for metric in metrics_names:

    mean, sd, ci_low, ci_high = calculate_statistics(
        participant_results[metric]
    )

    print(f"\n{metric}")

    print(
        f"{mean:.2f} ± {sd:.2f}%"
    )

    print(
        f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]"
    )


print("\nParticipant results:")
print(participant_results)
