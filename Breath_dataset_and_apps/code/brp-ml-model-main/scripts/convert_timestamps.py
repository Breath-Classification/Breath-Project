import os

import pandas as pd

folder_path = "../data/raw/"

for subdir in ["tens", "acc"]:
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(subdir_path, filename)

                df = pd.read_csv(file_path, delimiter=",", header=None)

                df[0] = pd.to_datetime(df[0])

                df[0] = (df[0] - df[0].min()).dt.total_seconds()

                df.columns = ["seconds", "data"]

                csv_filename = os.path.splitext(filename)[0] + ".csv"
                with open(os.path.join(subdir_path, csv_filename), "w") as f:
                    df.to_csv(f, index=False)

                os.remove(file_path)

print("Done!")
