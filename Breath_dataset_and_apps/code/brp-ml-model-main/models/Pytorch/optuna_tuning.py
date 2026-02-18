import optuna


from GRUMODELPYTROCH import GruModel
from GRUMODELPYTROCH import GRUAttentionModel
from LSTM.LSTM_Base import LSTM_BASE
from LSTM.LSTM_Dropout import LSTM_DROPOUT
from LSTM.LSTM_Stacked import LSTM_STACKED
from LSTM.LSTM_Bidirectional import LSTM_BIDIRECTIONAL
from LSTM.LSTM_Conv1 import LSTM_CONV1
from LSTM.LSTM_Attention import LSTM_ATTENTION
from Transformers.transformer import Transformer

from main import train_and_predict


def objective(trial):
    
    #hiperparametres

    lr =trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_units = trial.suggest_int("hidden_units", 32, 128)

    results =train_and_predict(block_size=30,batch_size=30,target=0,
                      hidden_units=hidden_units,output_shape=4,
                      model_type="LSTM_BASE",learning_rate=lr,
                      num_epchos=20)
    
    return max(results["max_test_acc"])


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3)
print("Najlepsze hiperparametry:", study.best_params)