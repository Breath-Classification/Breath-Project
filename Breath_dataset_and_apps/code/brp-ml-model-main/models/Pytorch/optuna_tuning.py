import optuna
from main import train_and_predict
from main import save_model


best_scores={}
def objective(trial,model_type):
    
    #hiperparametres

    lr =trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    hidden_units = trial.suggest_int("hidden_units", 64, 128)
    batch_size = trial.suggest_categorical("batch_size", [32,64,128])
    block_size = trial.suggest_int("block_size", 30,30)
    dropout = trial.suggest_float("dropout", 0.05, 0.6)
    num_layers =trial.suggest_int("num_layers", 1,3)
    if model_type == "Transformer":
            head_dim = trial.suggest_categorical("head_dim", [8,16,32,64])
            nhead = trial.suggest_categorical("nhead", [1,2,4,8])
            d_model = head_dim * nhead
            dim_feedforward = trial.suggest_categorical("dim_feedforward", [16,32,64,128])
    else:
            head_dim, nhead, d_model,dim_feedforward = None, None, None,None
    results =train_and_predict(block_size=block_size,batch_size=batch_size,target=0,
                      hidden_units=hidden_units,output_shape=4,
                      dataset_type="BlockDataset",
                      model_type=model_type,learning_rate=lr,
                      num_epchos=60,dropout=dropout,num_layers=num_layers,
                      dim_feedforward=dim_feedforward,
                      nhead=nhead,d_model=d_model, best_acc=best_scores.get(model_type, 0))
    
    acc = max(results["max_test_acc"])

    
    if acc > best_scores.get(model_type, 0):
        best_scores[model_type] = acc
    return acc

if __name__ == "__main__":
    
    models =[
        "LSTM_CONV1",
        "LSTM_DROPOUT",
        "LSTM_BASE",
        "LSTM_BIDIRECTIONAL",
        "LSTM_STACKED",
        "GruModel",
        "GRUAttentionModel",
   
    ]
    
    for model in models:
        best_scores[model] = 0.92
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial,model), n_trials=50)
        print("Najlepsze hiperparametry:", study.best_params)
    
    