from main import train_and_predict
from itertools import product
import json
import pathlib as PATH
TRIES = 5

def config_to_key(c):
    return str(c)


def latest_metric_value(values):
    if not values:
        return None
    return values[-1]

if __name__ == "__main__":
    
    grid = {
        "CRF": [True,False],
        "BIDIRECTIONAL": [True, False],
        "CONV1D" : [True,False],
        "DERIVATIVE": [True,False],
        "SLIDINGWINDOW" : [True,False],
        "PHYSICALCONSTRAINTS": [True,False]
    }

    keys = grid.keys()
    values = grid.values()
    
    config = [dict(zip(keys, v)) for v in product(*values)] # wszystkie kombinacje wartosci w gridzie
    
    result_dict = {}
    metric_names = [
    "max_test_acc",
    "max_standard_accuracy",
    "max_epsilon_accuracy",
    "max_transition_accuracy",
    "max_cycle_accuracy",
    "transition_edtt_f1",
    "transition_timing_mae",
    "min_avg_sizeof_error",
    "min_avg_position_error",
    "min_count_error",
]
    
    
    for c in config:
        print("RUN:", c)
        if c["SLIDINGWINDOW"] == False:
            dataset_type = "SequenceBlockDataset"
        else:
            dataset_type = "SequenceBlockWindowDataset"
        
        key = config_to_key(c)
        
        if key not in result_dict:
            result_dict[key] = {metric_name: [] for metric_name in metric_names}
            

        for i in range(TRIES):
            results = train_and_predict(
                block_size=30,
                batch_size=512,
                target=0,
                hidden_units=105,
                output_shape=4,
                model_type="LSTM_MIX",
                learning_rate=0.004393184532219237,
                num_epchos=70,
                dropout=0.3875032734956426,
                num_layers=2,
                dataset_type=dataset_type,
                loos_type="CrossEntropyLoss",
                lambda_con0=0.45253030622706797,
                lambda_con1=0.7980523096367567,
                lambda_con2=0.4705314248078593,
                lambda_con3=0.9098874037274234,
                use_CRF=c["CRF"],
                use_Physical=c["PHYSICALCONSTRAINTS"],
                use_bidirectional=c["BIDIRECTIONAL"],
                use_conv1d=c["CONV1D"],
                use_derivative=c["DERIVATIVE"],
                use_second_derivative=False)
            for metric_name in metric_names:
                result_dict[key][metric_name].append(latest_metric_value(results.get(metric_name, [])))
            with open("ablation_results.json", "w") as f:
                json.dump(result_dict, f, indent=4)
        with open("ablation_results2.json", "w") as f:
                json.dump(result_dict, f, indent=4)
    with open("ablation_results3.json", "w") as f:
                json.dump(result_dict, f, indent=4)   
        