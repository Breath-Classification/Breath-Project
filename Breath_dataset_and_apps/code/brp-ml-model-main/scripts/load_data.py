import csv
import re
from os import mkdir
from os.path import exists


def load_raw_data(filename: str) -> tuple[list[float], list[float]]:
    numbers = []
    times = []
    # Geting data from file
    with open(filename) as file:
        data = list(csv.reader(file))[1:]
        for data_line in data:
            numbers.append(float(data_line[1]))
            times.append(float(data_line[0]))
    return times, numbers


def load_tagged_data(filename: str) -> tuple[list[float], list[float]]:
    numbers = []
    tags = []
    # Geting data from file
    with open(filename) as file:
        data = file.read().splitlines()
        for data_line in data:
            numbers.append(float(data_line.split(",")[0]))
            tags.append(float(data_line.split(",")[1]))
    return numbers, tags


def load_data(filename):
    numbers = []
    # Geting data from file
    with open(filename) as file:
        data = file.read().splitlines()
        for data_line in data:
            match = re.search(r"y:(\d+\.\d+)", data_line)
            if match:
                number = float(match.group(1))
                numbers.append(number)

    return numbers[200:-200]


def save_data(filname, data):
    with open(filname, "w") as file:
        for element in data:
            file.write(f"{element[0]}\n")


def save_sequences(
    file_to_retrieve_sequences: str, file_to_save: str, size: int, certain_tags: list[int] | None = None,
) -> None:
    with open(file_to_retrieve_sequences):
        data, tags = load_tagged_data(file_to_retrieve_sequences)
    sequences = []
    for i in range(size, len(data)):
        position = i - size + size//2
        sequence = data[i - size: i]
        sequence.append(abs(max(sequence) - min(sequence)))
        if certain_tags is not None:
            if tags[position] in certain_tags:
                sequences.append(sequence + [tags[position] + 1])
        else:
            if tags[position] != 999.0:
                sequences.append(sequence + [tags[position] + 1])

    with open(file_to_save, "w") as file:
        for seq in sequences:
            file.write(",".join(map(str, seq)) + "\n")


def save_sequences_to_concatenated(
    file_to_retrieve_sequences: str, file_to_save: str
) -> None:
    with open(file_to_retrieve_sequences) as f:
        data = f.read().splitlines()

    with open(file_to_save, "a") as file:
        for seq in data:
            file.write(seq.strip() + "\n")


def empty_file(filename: str) -> None:
    folder = filename.split("/")[2]
    if not exists(f"data/pretrained/{folder}"):
        mkdir(f"data/pretrained/{folder}")

    with open(filename, "w"):
        pass


def prepare_data_for_training(sensor) -> None:
    input_size = sensor.value["size"] - 1
    sensor_type = sensor.value["name"]
    directory = f"data/pretrained"
    # directory2 = f"data/record_18-04-2024/pretrained"
    empty_file(f"{directory}/{sensor_type}_sequence/{sensor_type}_concatenated.txt")
    for data in [
        "_cough.txt",
        "_exhale_pause.txt",
        "_exhale_stop.txt",
        "_hyper.txt",
        "_inhale_pause.txt",
        "_inhale_stop.txt",
        "_normal.txt",
        "_shallow.txt",
        "_slow.txt",
        "_test.txt",
        "_yellow.txt",
        # "_second_subject.txt",
    ]:
        # save_sequences(
        #     f"{directory2}/{sensor_type}/{sensor_type}" + data,
        #     f"{directory2}/{sensor_type}_sequence/{sensor_type}" + data,
        #     input_size,
        # )
        save_sequences(
            f"{directory}/{sensor_type}/{sensor_type}" + data,
            f"{directory}/{sensor_type}_sequence/{sensor_type}" + data,
            input_size,
        )

        if data != "_test.txt":
            save_sequences_to_concatenated(
                f"{directory}/{sensor_type}_sequence/{sensor_type}" + data,
                f"{directory}/{sensor_type}_sequence/{sensor_type}_concatenated.txt",
            )
            # save_sequences_to_concatenated(
            #     f"{directory2}/{sensor_type}_sequence/{sensor_type}" + data,
            #     f"{directory}/{sensor_type}_sequence/{sensor_type}_concatenated.txt",
            # )
