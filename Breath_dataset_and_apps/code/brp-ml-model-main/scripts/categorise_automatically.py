import os

import numpy as np

from scripts.load_data import load_raw_data as load_data
from scripts.normalization import moving_average, normalize

# Load and normalize data
np.random.seed(42)
numbers = []
number_string = []
WINDOW_SIZE = 5

SENSOR_NAME = "acc"

current_directory = os.getcwd()
desired_directory = (
    os.path.dirname(os.path.dirname(current_directory))
    + f"/brp-ml-model/data/raw/{SENSOR_NAME}/"
)


def monotonicity(data, data_size=5) -> list[int]:
    deriv = np.gradient(data)
    result = [0] * len(data)

    for i in range(len(deriv) - data_size + 1):
        if abs(deriv[i]) < 0.0079:
            result[i] = 0
        elif deriv[i] > 0:
            result[i] = 1
        else:
            result[i] = -1
    return result


def check_monotonicity_change(subarray):
    increasing = False
    decreasing = False
    changes = 0
    x_points = np.arange(0, len(subarray))
    coefficients = np.polyfit(x_points, subarray, 1)
    a = coefficients[0]
    if max(subarray) - min(subarray) < 0.25:
        return 0
    for i in range(1, len(subarray)):
        if subarray[i] > subarray[i - 1]:
            if decreasing:
                changes += 1
                decreasing = False
            increasing = True
        elif subarray[i] < subarray[i - 1]:
            if increasing:
                changes += 1
                increasing = False
            decreasing = True

        if changes > 1:
            return 0

    return 1 if changes == 1 and -0.05 < a < 0.05 else 0


def find_monotonicity_changes(array, window_size=WINDOW_SIZE):
    results = []
    for i in range(window_size, len(array)):
        subarray = array[i - window_size : i]
        result = check_monotonicity_change(subarray)
        results.append(result)

    return results


def save_tagged_data(
    data: list[float], monotonicity: list[float], time: list[float], filename: str
) -> None:
    if not os.path.exists("../data/pretrained/"):
        os.mkdir("../data/pretrained/")
        os.mkdir("../data/pretrained/acc/")
        os.mkdir("../data/pretrained/tens/")

    if "tens" in filename:
        directory = "../data/pretrained/tens/"
    else:
        directory = "../data/pretrained/acc/"
    with open(directory + filename, "w") as file:
        for i in range(len(data)):
            file.write(f"{data[i]},{monotonicity[i]},{time[i]}\n")


for file in os.listdir(desired_directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        times, numbers = load_data(desired_directory + filename)

        if desired_directory[-4:] == "acc/":
            numbers = moving_average(numbers, 11)
            numbers = normalize(numbers, 375)

        else:
            numbers = moving_average(numbers, 5)
            numbers = normalize(numbers, 150)

        # if desired_directory[-4:] == "acc/":
        #     numbers = [-x for x in numbers]
        # Rotate numbers within the range of -1 to 1

        mono_tags = monotonicity(numbers, data_size=WINDOW_SIZE)

        save_tagged_data(numbers, mono_tags, times, filename[:-4] + ".txt")
