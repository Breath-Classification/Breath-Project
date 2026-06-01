import numpy as np


def normalize_window(window):
    min_val = min(window)
    max_val = max(window)
    range_val = max_val - min_val
    if range_val == 0:
        return [0 for _ in window]

    normalized = [(-1 + 2 * (x - min_val) / range_val) for x in window]
    return normalized


def normalize(numbers, normalization_range):
    normalized_values = []

    for i in range(len(numbers)):
        window = numbers[max(0, i - normalization_range) : i]
        try:
            normalized_window_values = normalize_window(window)
            normalized_values.append(normalized_window_values[-1])
        except ValueError:
            continue

    return normalized_values


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
