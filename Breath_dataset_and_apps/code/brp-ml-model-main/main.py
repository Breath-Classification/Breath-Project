import time

import numpy as np

from models.AbstractModel import SensorType
from models.Conv1DModel import Conv1DModel
from models.GRUModel import GRUModel
from models.LSTMModel import LSTMModel
from scripts.load_data import prepare_data_for_training
from scripts.plot import plot_evaluation_history, plot_history, plot_test_data
import matplotlib.pyplot as plt

# TODO: Change logic in labelling or any different
#  plot so it can be used in this test using predicted
#  data from model and X_test, y_test fields

SENSOR = SensorType.WIT_ACCELEROMETER
SENSOR_NAME = SENSOR.value["name"]

def evaluate_data_set_size():
    with open(f"data/pretrained/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt") as f:
        data = f.read().splitlines()
        tags = [float(seq.split(",")[-1]) for seq in data]

        # Define the number of bins
        num_bins = 4
        bin_counts, bin_edges = np.histogram(tags, bins=num_bins)

        # Prepare the bar plot
        bar_colors = ['red', 'green', 'blue', 'orange']  # Define different colors for each bar
        bar_labels = [f'Bin {i + 1}' for i in range(num_bins)]  # Optional labels for bins

        plt.bar(range(num_bins), bin_counts, color=bar_colors, tick_label=bar_labels)

        # Add counts on top of each rectangle
        for i, count in enumerate(bin_counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Bins')
        plt.ylabel('Number of Tags')
        plt.title('Tag Distribution Across Bins')
        plt.show()


def evaluate_epochs():
    prepare_data_for_training(sensor=SENSOR)
    history = []

    for i in range(10, 101, 10):
        _model = GRUModel()
        _model.load_data(
            filename=f"data/pretrained/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt",
            sensor_type=f"{SENSOR_NAME}",
        )
        _model.compile()
        _model.fit(epochs=i, sensor_type=f"{SENSOR_NAME}")
        history.append((i, _model.evaluate()))

    print(*history)

    plot_evaluation_history(history)


if __name__ == "__main__":
    prepare_data_for_training(sensor=SENSOR)
    # evaluate_data_set_size()
    scores = []
    model = GRUModel()
    model.load_data(
        filename=f"data/pretrained/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt",
        sensor_type=f"{SENSOR_NAME}",
    )
    # model.get_random_samples(1)
    model.compile()
    time_before = time.time()
    model.fit(sensor_type=f"{SENSOR_NAME}", epochs=100)
    print(f"Training time: {time.time() - time_before}")
    print(model.evaluate())
    # model = GRUModel()
    # model.load_data(
    #     filename=f"data/pretrained/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt",
    #     sensor_type=f"{SENSOR_NAME}",
    # )
    # model.get_random_samples(0.1)
    # model.compile()
    # time_before = time.time()
    # model.fit(sensor_type=f"{SENSOR_NAME}", epochs=100)
    # print(f"Training time: {time.time() - time_before}")
    # model.load(f"models/saves/{SENSOR_NAME}/GRUModel_{SENSOR_NAME}")
    model.save(f"models/saves/{SENSOR_NAME}/GRUModel_{SENSOR_NAME}")
    model.confusion_matrix(model.X_test, model.y_test, name=f"{SENSOR_NAME}_test")
    model.plot_prediction(model.X_test, name=f"{SENSOR_NAME}_test")
    plot_history(f"models/saves/{SENSOR_NAME}/GRUModel.history")
    # plot_test_data(SENSOR_NAME, normalize_data=True)
