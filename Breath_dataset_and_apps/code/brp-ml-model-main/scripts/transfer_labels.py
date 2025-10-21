import os
import glob

TIME_TOLERANCE = 0.1

def parse_line(line):
    value, tag, second = line.strip().split(',')
    return value, tag, float(second)

def format_line(value, tag, second):
    """Formats the value, tag, and second back into a line."""
    return f"{value},{tag},{second}\n"

def transfer_label(tens_folder, acc_folder):
    tens_files = glob.glob(os.path.join(tens_folder, "tens_*.txt"))
    acc_files = glob.glob(os.path.join(acc_folder, "acc_*.txt"))

    # Normalize file paths
    acc_files = [os.path.normpath(f) for f in acc_files]


    for tens_file in tens_files:
        tens_data = []

        # Read data from the tens file
        with open(tens_file, 'r') as f:
            for line in f:
                tens_data.append(parse_line(line))

        print(tens_data)
        # Derive the corresponding acc file name
        file_name_suffix = os.path.basename(tens_file).replace("tens_", "")
        acc_file = os.path.join(acc_folder, f"acc_{file_name_suffix}")
        acc_file = os.path.normpath(acc_file)

        # Debug: print the expected acc file path

        if acc_file not in acc_files:
            continue

        # Read and process the acc file
        acc_data = []
        with open(acc_file, 'r') as f:
            for line in f:
                acc_value, acc_tag, acc_second = parse_line(line)

                # Find the closest matching timestamp in tens data
                closest_match = None
                closest_diff = float('inf')
                for tens_value, tens_tag, tens_second in tens_data:
                    time_diff = abs(tens_second - acc_second)
                    if time_diff < closest_diff and time_diff <= TIME_TOLERANCE:
                        closest_diff = time_diff
                        closest_match = tens_tag

                # Replace the tag in acc data if a match was found
                if closest_match is not None:
                    acc_tag = closest_match

                acc_data.append(format_line(acc_value, acc_tag, acc_second))

        # Write the modified data back to the acc file
        with open(acc_file, 'w') as f:
            f.writelines(acc_data)

if __name__ == "__main__":
    tens_folder = os.path.normpath("../data/pretrained/tens")
    acc_folder = os.path.normpath("../data/pretrained/acc")
    transfer_label(tens_folder, acc_folder)

