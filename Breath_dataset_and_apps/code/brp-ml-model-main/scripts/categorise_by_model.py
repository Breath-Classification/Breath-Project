import os

import numpy as np

from models.AbstractModel import SensorType
from models.GRUModel import GRUModel
from scripts.load_data import load_raw_data as load_data
from scripts.normalization import moving_average, normalize

SENSOR = SensorType.TENSOMETER
SENSOR_NAME = SENSOR.value["name"]
WINDOWS_SIZE = SENSOR.value["size"]

current_directory = os.getcwd()
desired_directory = (
    os.path.dirname(os.path.dirname(current_directory))
    + f"/brp-ml-model/data/raw/{SENSOR_NAME}/"
)

def predict_tags(data: list[float], window_size: int) -> list[int]:
    model = GRUModel()
    model.load(f"../models/saves/{SENSOR_NAME}/GRUModel_{SENSOR_NAME}")

    tags = []
    data_to_predict = []
    for i in range(len(data)-5):
        numbers = data[i:i+5]
        numbers.extend([abs(max(data[i:i+5]) - min(data[i:i+5]))])
        data_to_predict.append(numbers)
    tags.append(model.predict(np.array(data_to_predict)))
    return tags[0]

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
            file.write(f"{data[i]},{monotonicity[i]-1},{time[i]}\n")


for file in os.listdir(desired_directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        times, numbers = load_data(desired_directory + filename)

        if desired_directory[-4:] == "acc/":
            numbers = moving_average(numbers, WINDOWS_SIZE)
            numbers = normalize(numbers, 150)

        else:
            numbers = moving_average(numbers, WINDOWS_SIZE)
            numbers = normalize(numbers, 150)

        # if desired_directory[-4:] == "acc/":
        #     numbers = [-x for x in numbers]
        # Rotate numbers within the range of -1 to 1

        mono_tags = predict_tags(numbers, WINDOWS_SIZE)

        save_tagged_data(numbers[:-WINDOWS_SIZE], mono_tags, times[:-WINDOWS_SIZE], filename[:-4] + ".txt")