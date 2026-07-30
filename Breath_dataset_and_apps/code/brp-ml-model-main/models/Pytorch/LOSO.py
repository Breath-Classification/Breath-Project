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

                    if line:  # pomijamy puste linie
                        outfile.write(line + "\n")

    return output_file


if __name__ == "__main__":

    folds_dir = Path("../../data/NewData/sequence/Folds")
    results_dir = Path("results")

    results_dir.mkdir(exist_ok=True)


    folds = sorted(
        [x for x in folds_dir.iterdir() if x.is_dir()]
    )


    for fold in folds:

        print(f"\n===== {fold.name} =====")


        # tworzenie concatenated
        train_file = concatenate_sequences(
            fold / "train"
        )

        test_file = concatenate_sequences(
            fold / "test"
        )


        # trening
        results = train_and_predict(
            block_size=30,
            batch_size=32,
            target=0,
            hidden_units=106,
            output_shape=4,
            model_type="LSTM_MIX",
            learning_rate=0.003461279782396843,
            num_epchos=100,
            dropout=0.3875032734956426,
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


        # zapis wyników
        result_file = results_dir / f"{fold.name}_results.json"

        with open(result_file, "w") as f:
            json.dump(
                results,
                f,
                indent=4,
                default=str
            )


        print(f"{fold.name} zapisany.")
        
        