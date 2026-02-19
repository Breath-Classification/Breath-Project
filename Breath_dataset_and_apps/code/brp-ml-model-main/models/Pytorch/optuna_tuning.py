import optuna
from main import train_and_predict


def objective(trial):
    
    #hiperparametres

    lr =trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    hidden_units = trial.suggest_int("hidden_units", 32, 128)
    batch_size = trial.suggest_categorical("batch_size", [16,32,64,128])
    block_size = trial.suggest_int("block_size", 25,35)
    dropout = trial.suggest_float("dropout", 0.05, 0.8)
    num_layers =trial.suggest_int("num_layers", 1,5)
    head_dim = trial.suggest_categorical("head_dim", [8,16,32,64])
    nhead = trial.suggest_categorical("nhead", [1,2,4,8])
    d_model = head_dim *nhead
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [16,32,64,128])
    results =train_and_predict(block_size=block_size,batch_size=batch_size,target=0,
                      hidden_units=hidden_units,output_shape=4,
                      model_type="Transformer",learning_rate=lr,
                      num_epchos=100,dropout=dropout,num_layers=num_layers,dim_feedforward=dim_feedforward,nhead=nhead,d_model=d_model)
    
    return max(results["max_test_acc"])


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Najlepsze hiperparametry:", study.best_params)