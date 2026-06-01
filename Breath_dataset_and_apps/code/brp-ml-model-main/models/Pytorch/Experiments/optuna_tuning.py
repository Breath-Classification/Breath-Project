import optuna
import os
from main import train_and_predict


best_scores={}
def build_score_key(model_type, loss, dataset_type):
    return f"{model_type}|{loss}|{dataset_type}"
def build_save_location(model_type, loss, dataset_type):
    base_path = os.path.join(
        "models",
        "saved_models",
        "optuna",
        dataset_type,
        loss,
    )
    filename = f"{model_type}_best"
    return base_path, filename
def objective(trial,model_type,loss,dataset_type):
    
    #hiperparametres

    lr =trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    hidden_units = trial.suggest_int("hidden_units", 64, 128)
    batch_size = trial.suggest_categorical("batch_size", [32,64,128,256,512,1024])
    block_size = trial.suggest_int("block_size", 30,30)
    dropout = trial.suggest_float("dropout", 0.05, 0.6)
    num_layers =trial.suggest_int("num_layers", 1,3)
    #lambda_con0 = trial.suggest_float("lambda_con0",0.00, 1.00)
    #lambda_con1 = trial.suggest_float("lambda_con1",0.00, 1.00)
    #lambda_con2 = trial.suggest_float("lambda_con2",0.00, 1.00)
    #lambda_con3 = trial.suggest_float("lambda_con3",0.00, 1.00)
    if model_type == "Transformer":
            head_dim = trial.suggest_categorical("head_dim", [8,16,32,64])
            nhead = trial.suggest_categorical("nhead", [1,2,4,8])
            d_model = head_dim * nhead
            dim_feedforward = trial.suggest_categorical("dim_feedforward", [16,32,64,128])
    else:
            head_dim, nhead, d_model,dim_feedforward = None, None, None,None
    save_path, save_filename = build_save_location(model_type, loss, dataset_type)
    score_key = build_score_key(model_type, loss, dataset_type)
    results =train_and_predict(block_size=block_size,batch_size=batch_size,target=0,
                      hidden_units=hidden_units,output_shape=4,
                      dataset_type=dataset_type,
                      model_type=model_type,learning_rate=lr,
                      num_epchos=50,dropout=dropout,num_layers=num_layers,
                      dim_feedforward=dim_feedforward,
                      nhead=nhead,d_model=d_model, best_acc=best_scores.get(score_key, 0),
                      loos_type=loss,
                      save_path=save_path,
                      save_filename=save_filename
                      #lambda_con0=lambda_con0,lambda_con1=lambda_con1,
                      #lambda_con2=lambda_con2,lambda_con3=lambda_con3
                      )
    
    acc = max(results["max_test_acc"])

    
    if acc > best_scores.get(score_key, 0):
        best_scores[score_key] = acc
    return acc

if __name__ == "__main__":
    
    models =[
        "LSTM_ATTENTION",   
        "LSTM_STACKED",
        "GruModel",
        "LSTM_BIDIRECTIONAL",
        "GRUAttentionModel",
        "LSTM_BASE",
        "LSTM_DROPOUT",
        "LSTM_CONV1",
        "Transformer"
    ]
    loss = [
         "FocalLoss",
         "FocalLossAdaptive",
    ]
    dataset = [
         "BlockDataset"
    ]
    
    
   
    for model in models:
        for los in loss:
            for data in dataset:
                if model == "LSTM_ATTENTION" and los == "FocalLoss":
                    continue
                score_key = build_score_key(model, los, data)
                best_scores[score_key] = 0.87
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: objective(trial,model,los,data), n_trials=70)
                print("Najlepsze hiperparametry:", study.best_params)
    
    