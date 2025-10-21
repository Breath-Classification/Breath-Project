from multiprocessing import Process, Queue, freeze_support
from tkinter import Button, Frame, Tk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector

matplotlib.use("TkAgg")

# Load data from txt file
FILENAME = "yellow"
tens_file_path = f"../data/pretrained/tens/tens_{FILENAME}.txt"
acc_file_path = f"../data/pretrained/acc/acc_{FILENAME}.txt"
tens_data = np.loadtxt(tens_file_path, delimiter=",")
acc_data = np.loadtxt(acc_file_path, delimiter=",")

# Constants
COLOR_MAP = {-1.0: "red", 0.0: "green", 1.0: "blue", 2.0: "orange", 999.0: "black"}
SECONDS_TIME = 15
SELECTION_TIME_SIZE = 10

# Global variables
current_start_index_tens = 0
current_start_index_acc = 0
span_selector_active_tens = False
span_selector_active_acc = False


def get_data_according_to_time(data, time_start, time_stop):
    return data[(data[:, 2] >= time_start) & (data[:, 2] < time_stop)]


def get_data_indexes_according_to_time(data, time_start, time_stop):
    start_index = np.where(data[:, 2] >= time_start)[0][0]
    end_index = np.where(data[:, 2] <= time_stop)[0][-1]
    return start_index, end_index


# Interactive plotting
def plot_data(ax, data, current_start_index, title):
    ax.cla()  # Clear current axes
    end_index = current_start_index + SECONDS_TIME
    gather_time_relevant_data = get_data_according_to_time(
        data, current_start_index, end_index
    )
    colors = [COLOR_MAP[label] for label in gather_time_relevant_data[:, 1]]
    scatter = ax.scatter(
        gather_time_relevant_data[:, 2],
        gather_time_relevant_data[:, 0],
        color=colors,
        picker=True,
    )
    ax.plot(
        gather_time_relevant_data[:, 2],
        gather_time_relevant_data[:, 0],
        linestyle="-",
        color="gray",
        alpha=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("Normalized Value")
    ax.grid(True)
    fig.canvas.draw()
    scatter.set_picker(True)


def onpick(event, data, current_start_index, file_path):
    if (
        not (span_selector_active_tens or span_selector_active_acc)
        and event.artist.get_label() == "_child0"
    ):
        point_index = int(event.ind[0] + current_start_index)
        current_color = int(data[point_index, 1])
        new_color = choose_color(current_color)
        update_point_color(point_index, point_index + 1, new_color, data, file_path)


def update_point_color(start_index, end_index, new_color, data, file_path):
    # Update color in the data array
    for i in range(start_index, end_index):
        data[i, 1] = new_color
    plot_data(ax1, tens_data, current_start_index_tens, "Tens Plot")
    plot_data(ax2, acc_data, current_start_index_acc, "Acc Plot")

    # Save updated data to the txt file
    np.savetxt(file_path, data, fmt="%.18f", delimiter=",")


def choose_color(initial_color):
    result = {"color": None}

    def select_color(color):
        result["color"] = color
        root.destroy()

    root = Tk()
    root.title("Choose Color")

    frame = Frame(root)
    frame.pack(padx=10, pady=10)
    BREATH_STATE = {
        -1.0: "BREATH OUT",
        0.0: "OUT NO BREATH",
        1.0: "BREATH IN",
        2.0: "IN NO BREATH",
        999.0: "NO RELEVANT"
    }
    for label, color in COLOR_MAP.items():
        button_label = BREATH_STATE[label]
        button = Button(
            frame,
            text=str(button_label),
            command=lambda c=label: select_color(c),
            bg=color,
            width=8,
            highlightbackground=color,
        )
        button.pack(side="left", padx=5)

    root.protocol("WM_DELETE_WINDOW", lambda: select_color(initial_color))
    root.wait_window()

    return result["color"]


def on_key(event, data, current_start_index, ax):
    global \
        current_start_index_tens, \
        current_start_index_acc, \
        span_selector_active_tens, \
        span_selector_active_acc
    if event.key == "right":
        if ax == ax1:
            current_start_index_tens += SELECTION_TIME_SIZE
            if current_start_index_tens > len(data) - SECONDS_TIME:
                current_start_index_tens = len(data) - SECONDS_TIME
        elif ax == ax2:
            current_start_index_acc += SELECTION_TIME_SIZE
            if current_start_index_acc > len(data) - SECONDS_TIME:
                current_start_index_acc = len(data) - SECONDS_TIME
    elif event.key == "left":
        if ax == ax1:
            current_start_index_tens -= SELECTION_TIME_SIZE
            if current_start_index_tens < 0:
                current_start_index_tens = 0
        elif ax == ax2:
            current_start_index_acc -= SELECTION_TIME_SIZE
            if current_start_index_acc < 0:
                current_start_index_acc = 0
    elif event.key == "m":
        # Toggle span selector mode
        if ax == ax1:
            span_selector_active_tens = not span_selector_active_tens
            if span_selector_active_tens:
                span_selector_tens.set_active(True)
            else:
                span_selector_tens.set_active(False)
        elif ax == ax2:
            span_selector_active_acc = not span_selector_active_acc
            if span_selector_active_acc:
                span_selector_acc.set_active(True)
            else:
                span_selector_acc.set_active(False)

    plot_data(ax1, tens_data, current_start_index_tens, "Tens Plot")
    plot_data(ax2, acc_data, current_start_index_acc, "Acc Plot")


def on_span_select(
    xmin,
    xmax,
    data_tens,
    data_acc,
    current_start_index,
    ax,
    tens_file_path,
    acc_file_path,
):
    start_index = max(xmin, 0)
    end_index = min(xmax, max(data_tens[:, 2]))
    data_start, data_end = get_data_indexes_according_to_time(
        data_tens, start_index, end_index
    )
    new_color = choose_color(int(data_tens[data_start, 1]))
    update_point_color(data_start, data_end, new_color, data_tens, tens_file_path)

    data_start, data_end = get_data_indexes_according_to_time(
        data_acc, start_index, end_index
    )
    update_point_color(data_start, data_end, new_color, data_acc, acc_file_path)


def on_span_select_acc(
        xmin,
        xmax,
        data_tens,
        data_acc,
        current_start_index,
        ax,
        acc_file_path,
):
    start_index = max(xmin, 0)
    end_index = min(xmax, max(data_acc[:, 2]))
    data_start, data_end = get_data_indexes_according_to_time(
        data_acc, start_index, end_index
    )
    new_color = choose_color(int(data_acc[data_start, 1]))
    update_point_color(data_start, data_end, new_color, data_acc, acc_file_path)


def tk_process(q):
    plt.show()
    q.put("done")


# Create initial plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
# Connect the pick event to the function
fig.canvas.mpl_connect(
    "pick_event",
    lambda event: onpick(event, tens_data, current_start_index_tens, tens_file_path),
)
fig.canvas.mpl_connect(
    "key_press_event",
    lambda event: on_key(event, tens_data, current_start_index_tens, ax1),
)
fig.canvas.mpl_connect(
    "pick_event",
    lambda event: onpick(event, acc_data, current_start_index_acc, acc_file_path),
)
fig.canvas.mpl_connect(
    "key_press_event",
    lambda event: on_key(event, acc_data, current_start_index_acc, ax2),
)

plot_data(ax1, tens_data, current_start_index_tens, "Tens Plot")
plot_data(ax2, acc_data, current_start_index_acc, "Acc Plot")
# Create a SpanSelector for tens plot
span_selector_tens = SpanSelector(
    ax1,
    lambda xmin, xmax: on_span_select(
        xmin,
        xmax,
        tens_data,
        acc_data,
        current_start_index_tens,
        ax1,
        tens_file_path,
        acc_file_path,
    ),
    "horizontal",
    useblit=True,
)
span_selector_tens.set_active(False)

# Create a SpanSelector for acc plot
span_selector_acc = SpanSelector(
    ax2,
    lambda xmin, xmax: on_span_select_acc(
        xmin,
        xmax,
        tens_data,
        acc_data,
        current_start_index_acc,
        ax2,
        acc_file_path,
    ),
    "horizontal",
    useblit=True,
)
span_selector_acc.set_active(False)

# Ensure that the script runs in the main block when using multiprocessing
if __name__ == "__main__":
    # Add freeze_support() before starting any process
    freeze_support()

    # Create a queue to communicate between the processes
    q = Queue()

    # Start the Tkinter process
    tkinter_process = Process(target=tk_process, args=(q,))
    tkinter_process.start()

    # Wait for the Tk
    q.get()
