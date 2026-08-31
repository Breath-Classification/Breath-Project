from pathlib import Path
import json

from main import train_and_predict


def concatenate_sequences(folder):

    output_file = folder / "concatenated.txt"

    sequence_files = sorted(folder.glob("*_sequence.txt"))

    with open(output_file, "w") as outfile:

        for file in sequence_files:

            with open(file, "r") as infile:

                for line in infile:

                    line = line.strip()

                    if line:
                        outfile.write(line + "\n")

    return output_file


if __name__ == "__main__":

    folds_dir = Path("../../data/NewData/sequence/Folds")
    results_dir = Path("results")

    results_dir.mkdir(exist_ok=True)


    folds = sorted(
        [x for x in folds_dir.iterdir() if x.is_dir()]
    )


    NUM_RUNS = 5


    for fold in folds:

        print(f"\n===== {fold.name} =====")


        # folder dla konkretnego folda
        fold_results_dir = results_dir / fold.name
        fold_results_dir.mkdir(exist_ok=True)


        # tworzenie concatenated
        train_file = concatenate_sequences(
            fold / "train"
        )

        test_file = concatenate_sequences(
            fold / "test"
        )


        for run in range(1, NUM_RUNS + 1):

            print(
                f"\n--- {fold.name} RUN {run}/{NUM_RUNS} ---"
            )


            results = train_and_predict(
                block_size=30,
                batch_size=32,
                target=0,
                hidden_units=106,
                output_shape=4,
                model_type="LSTM_MIX",
                learning_rate=0.003461279782396843,
                num_epchos=100,
                dropout=0.3875033062276,
                num_layers=2,

                lambda_con0=0.45253030622706797,
                lambda_con1=0.7980523096367567,
                lambda_con2=0.4705314248078593,
                lambda_con3=0.9098874037274234,

                dataset_type="SequenceBlockWindowDataset",
                loos_type="CrossEntropyLoss",

                train_data_path=str(train_file),
                test_data_path=str(test_file),
            )


            result_file = (
                fold_results_dir /
                f"run_{run}_results.json"
            )


            with open(result_file, "w") as f:

                json.dump(
                    results,
                    f,
                    indent=4,
                    default=str
                )


            print(
                f"{fold.name} run {run} zapisany."
            )