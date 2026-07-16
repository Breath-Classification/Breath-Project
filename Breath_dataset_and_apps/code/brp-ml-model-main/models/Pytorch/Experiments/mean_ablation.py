import json
from statistics import mean

# Wczytaj plik JSON
with open("ablation_results3.json", "r") as f:
    data = json.load(f)

# Słownik na wyniki
averaged_results = {}

for run_name, metrics in data.items():
    averaged_results[run_name] = {}

    for metric_name, values in metrics.items():
        if isinstance(values, list):
            averaged_results[run_name][metric_name] = mean(values)
        else:
            averaged_results[run_name][metric_name] = values

# Zapisz do nowego pliku
with open("results_averaged.json", "w") as f:
    json.dump(averaged_results, f, indent=4)

print("Średnie zostały zapisane do results_averaged.json")