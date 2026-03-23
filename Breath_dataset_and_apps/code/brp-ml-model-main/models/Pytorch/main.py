#data_loader
from data_loader import create_dataloaders
from torchvision import transforms
#models
from models.GRU.GruModel import GruModel
from models.GRU.GruAttention import GRUAttentionModel
from models.GRU.GruSeq import Seq2SeqGRU
from models.LSTM.LSTM_Base import LSTM_BASE
from models.LSTM.LSTM_Dropout import LSTM_DROPOUT
from models.LSTM.LSTM_Stacked import LSTM_STACKED
from models.LSTM.LSTM_Bidirectional import LSTM_BIDIRECTIONAL
from models.LSTM.LSTM_Conv1 import LSTM_CONV1
from models.LSTM.LSTM_Attention import LSTM_ATTENTION
from models.LSTM.LSTM_Mix import LSTM_MIX
from models.Transformers.transformer import Transformer
from models.Transformers.transformer_CNN_CRF import Transformer_CNN_CRF
#loss functions
from Weightening.focal_loss import FocalLoss
from Weightening.adaptive_focal_loss import FocalLossAdaptive
#engines
import Engines.engine
import Engines.engine_CRF
import Engines.engine_without_epsilon
#libraries
import torch
import wandb 
from enum import Enum
import os
import json
#scripts
from scripts.HMM import viterbi_algorithm 


#Constants
class SensorType(Enum):
    TENSOMETER = {"name": "tens", "size": 6}
    ACCELEROMETER = {"name": "acc", "size": 12}
    WIT_ACCELEROMETER = {"name": "acc", "size": 12}    
NUM_EPOCHS = 20
LEARNING_RATE = 0.01 #dla LSTM 0.001
BATCHES = 32
BLOCK_SIZE=[30]
SENSOR = SensorType.TENSOMETER
SENSOR_NAME = SENSOR.value["name"]
TARGET = 2

#Usage of Wandb


#use_wandb = input("Włączyć W&B? (y/n): ").strip().lower()
use_wandb='n'
print (use_wandb)
if use_wandb != "y":
    os.environ["WANDB_DISABLED"] = "true"


def train_and_predict(block_size,batch_size,target,hidden_units,output_shape,
                      model_type,learning_rate,num_epchos,
                      dataset_type="SequenceDataset", 
                      loos_type ="CrossEntropyLoss", optimizer_type="Adam",
                      dropout =0, num_layers=2, dim_feedforward =64, 
                      nhead  =2, d_model=32,best_acc=0.99):
   
    #logs
    wandb.init(
        mode="online" if  use_wandb=='y' else "disabled",
        project="GRU-optymalization",
        name=f"Transformer {block_size} ",
        group="Transformers ",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCHES,
            "lr": LEARNING_RATE,
            "block_size":BLOCK_SIZE,
            "model":"Transformer",
            "sensor":SENSOR_NAME,
            "loss_fn":"CrossEntropyLoss",
            "optimizer":"Adam"
        }
    ) 
    train,test =config_dataloaders(block_size,batch_size,target,dataset_type)

    model = create_model(hidden_units,output_shape,model_type,train,test,dropout,num_layers,dim_feedforward,nhead, d_model)
    model.to("cuda")
    loss_fn = create_loos_function(loos_type)
    optimizer = create_optimizer(optimizer_type,model,learning_rate)

    if dataset_type=="BlockDataset":
        results,end =Engines.engine.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", True, best_acc) #stop i set 
    elif dataset_type=="SequenceBlockDataset":
        results =Engines.engine_CRF.train(model, train, test, optimizer, loss_fn, num_epchos, "cpu", False, 0.942) #stop i set
    elif  dataset_type=="SequenceDataset":
        results,end =Engines.engine.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", False, 0.942) #stop i set
    elif  dataset_type=="SequenceBlockWindowDataset":
        results =Engines.engine_CRF.train(model, train, test, optimizer, loss_fn, num_epchos, "cuda", True, 0.90) #stop i set  
    
    wandb.finish()

    #optuna tuning 
    #return results
    
    
    config = {
        "block_size": block_size,
        "batch_size": batch_size,
        "target": target,
        "hidden_units": hidden_units,
        "output_shape": output_shape,
        "model_type": model_type,
        "learning_rate": learning_rate,
        "num_epochs": num_epchos,
        "loos_type":loos_type,
        "optimizer_type":optimizer_type,
        "dataset_type":dataset_type
    }
    end=True
    if end ==True:
        save_model(model,config,path="models/saved_models/",filename=f"{model_type}_S{best_acc:.4f}")
    #save_model_mobile(model)
    return results
    
    

    all_preds, all_trues, all_features,_ = evaluate_model(model,test,dataset_type)
    return all_preds, all_trues, all_features
def save_model_mobile(model, filename="LSTM_BaseMobile"):
    # Przełącz model w tryb ewaluacji
    model.eval()
    
    # Tworzymy przykładowy tensor wejściowy (musisz dopasować do swojego inputu)
    example_input = torch.rand(1, 30, 6)  # np. batch=1, 30x6 wartości
    
    # Tworzymy TorchScript
    #traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module = torch.jit.script(model)
    # Zapisujemy w folderze 'mobile_models'
    import os
    os.makedirs("models/mobile_models", exist_ok=True)
    traced_script_module.save(f"models/mobile_models/{filename}.pt")
    print(f"TorchScript model saved as models/mobile_models/{filename}.pt")
    
def save_model(model, config, path, filename=""):
    #saving model
    if use_wandb == 'n':
        if filename=="":
            filename = input("input name of the saved model ")
        torch.save(model.state_dict(), f"{path}/{filename}.pth")
    else:
        torch.save(model.state_dict(), f"{path}/Class_Weightening.pth")
    
    with open(f"{path}/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def evaluate_model(model, test,dataset_type):
    all_preds = []
    all_trues = []

    all_features = []
    model.eval()
    return_sequence = False # USED FOR HMM
    
    if(not return_sequence):
        if dataset_type=="BlockDataset":
            with torch.no_grad():
                for X, y in test:
                    y_pred = model(X)
                
                    pred_classes = torch.argmax(y_pred, dim=1) 
                    all_preds.append(pred_classes.cpu())
                    all_trues.append(y.cpu())
                    all_features.append(X.cpu())
                all_preds =torch.cat(all_preds)
                all_trues =torch.cat(all_trues)
                all_features =torch.cat(all_features)
        
                
                all_features = all_features[:, -1, :6]
                all_features = all_features.mean(dim=1)

                
                return all_preds, all_trues, all_features,0
        elif dataset_type == "SequenceBlockDataset":
            with torch.no_grad():
                for X, y in test:
                    y_pred = model(X)
                
                    pred_classes = torch.argmax(y_pred, dim=2)  
                    all_preds.append(pred_classes.cpu())
                    all_trues.append(y.cpu())
                    all_features.append(X.cpu())
                all_preds =torch.cat(all_preds)
                all_trues =torch.cat(all_trues)
                all_features =torch.cat(all_features)
        
                all_features = all_features[:, :, :6] 
                all_features = all_features.reshape(-1, all_features.shape[2]) 
                all_features = all_features.mean(dim=1)
                
                all_features = all_features.squeeze(-1)
                all_features = all_features.flatten()
                all_trues = all_trues.flatten()
                all_preds = all_preds.flatten() 
                
                return all_preds, all_trues, all_features,0
        elif dataset_type == "SequenceDataset":
            with torch.no_grad():
                for X, y in test:
                    y_pred = model(X)

                    pred_classes = torch.argmax(y_pred, dim=1)
                    all_preds.append(pred_classes.cpu())
                    all_trues.append(y.cpu())
                    all_features.append(X.cpu())
                all_preds = torch.cat(all_preds)
                all_trues = torch.cat(all_trues)
                all_features = torch.cat(all_features)

                # SequenceDataset can be:
                # - [N, F] when expand_dims=False
                # - [N, 1, F] when expand_dims=True
                if all_features.dim() == 3:
                    all_features = all_features[:, 0, :6]
                elif all_features.dim() == 2:
                    all_features = all_features[:, :6]
                else:
                    raise ValueError(f"Unexpected SequenceDataset feature shape: {all_features.shape}")
                all_features = all_features.mean(dim=1)

                all_features = all_features.squeeze(-1)
                all_features = all_features.flatten()
                all_trues = all_trues.flatten()
                all_preds = all_preds.flatten()

                return all_preds, all_trues, all_features,0
        elif dataset_type == "SequenceBlockWindowDataset":
            with torch.no_grad():
                for X, y in test:
                    y_pred = model(X)

                    pred_classes = torch.argmax(y_pred, dim=2)
                    all_preds.append(pred_classes.cpu())
                    all_trues.append(y.cpu())
                    all_features.append(X.cpu())
                all_preds = torch.cat(all_preds)
                all_trues = torch.cat(all_trues)
                all_features = torch.cat(all_features)

                all_features = all_features[:, :, :6]
                all_features = all_features.reshape(-1, all_features.shape[2])
                all_features = all_features.mean(dim=1)

                all_features = all_features.squeeze(-1)
                all_features = all_features.flatten()
                all_trues = all_trues.flatten()
                all_preds = all_preds.flatten()

                return all_preds, all_trues, all_features,0
            
    else:
        print("Uzywam HMM")
        all_paths = []
        with torch.no_grad():
            for X, y in test:
                y_pred = model(X)  
                pred_classes = torch.argmax(y_pred, dim=1)  
                all_preds.append(pred_classes.cpu())
                all_trues.append(y.cpu())
                all_features.append(X.cpu())

                output = model(X, return_sequence=True)   # (batch, T, hidden)
                logits = model.fc(output)                 # (batch, T, 4)
                log_probs = torch.log_softmax(logits, -1)
                
                for i in range(BATCHES):
                    path = viterbi_algorithm(log_probs[i])
                    last_label = torch.tensor(path[-1]) 
                    all_paths.append(last_label.cpu())
                    
                #print("output shape:", output.shape)  
                #print("logits shape:", logits.shape) 
                #print("log_probs shape:", log_probs.shape)
        all_preds =torch.cat(all_preds)
        all_trues =torch.cat(all_trues)
        all_features =torch.cat(all_features)
        all_features = all_features.mean(dim=1) 
        all_paths = torch.tensor(all_paths)
        print(len(all_paths), all_paths.shape)
        print(len(all_preds), all_preds.shape)
        print(len(all_trues), all_trues.shape)
        return all_preds, all_trues, all_features, all_paths
        
def create_loos_function(loss_type):
    if loss_type == "FocalLossAdaptive":
        loss_fn = FocalLossAdaptive(gamma=2)
    elif loss_type == "FocalLoss":
        loss_fn = FocalLoss(gamma=2)
    elif loss_type =="CrossEntropyLoss":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("wrong loss_type")
    return loss_fn

def create_optimizer(optimizer_type, model, learning_rate):
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("wrong optimizer_type")
    return optimizer


def config_dataloaders(block_size,batch_size,target,dataset_type):
    data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    train, test = create_dataloaders(transform=data_transform,
                                         batch_size=batch_size,
                                         block_size=block_size,
                                         target=target,
                                         dataset_type=dataset_type)

    return train,test

def create_model(hidden_units,output_shape,model_type,train,test, dropout=0, num_layers=2, dim_feedforward =64, nhead  =2, d_model=32):
    
    X_batch, y_batch = next(iter(train))
   
    input_shape = X_batch.shape[-1]
   

    if use_wandb == 'n':
        if model_type == "LSTM_ATTENTION":
            model = LSTM_ATTENTION(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape,
                                    dropout=dropout,
                                    num_layers=num_layers)
        elif model_type == "LSTM_STACKED":
            model = LSTM_STACKED(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_type == "GruModel":
            model = GruModel(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_type == "LSTM_BIDIRECTIONAL":
            model = LSTM_BIDIRECTIONAL(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_type == "GRUAttentionModel":
            model = GRUAttentionModel(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_type == "LSTM_BASE":
            model = LSTM_BASE(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_type == "LSTM_DROPOUT":
            model = LSTM_DROPOUT(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_type == "LSTM_CONV1":
            model = LSTM_CONV1(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_type == "Transformer":
            model = Transformer(input_shape=input_shape,
                                    d_model=d_model,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape,
                                    dropout=dropout,
                                    num_layrer=num_layers,
                                    dim_feedforward=dim_feedforward,
                                    nhead=nhead,
                                    )
        elif model_type == "LSTM_MIX":
            model = LSTM_MIX(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape,
                                    dropout=dropout,
                                    num_layers=num_layers)
        elif model_type == "Transformer_CNN_CRF":
            model = Transformer_CNN_CRF(input_shape=input_shape,
                                    d_model=d_model,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape,
                                    dropout=dropout,
                                    num_layrer=num_layers,
                                    dim_feedforward=dim_feedforward,
                                    nhead=nhead,
                                    )
        else:
            raise ValueError("wrong model_type")
    else:
        model = Transformer(input_shape=input_shape,d_model=32,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
    return model
    

def load_model_and_predict(model_path):

    filename = model_path[:-4]
    with open(f"{filename}.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    
    block_size =config["block_size"]
    batch_size =config["batch_size"]
    target =config["target"]
    hidden_units =config["hidden_units"]
    output_shape =config["output_shape"]
    model_type =config["model_type"]
    learning_rate =config["learning_rate"]
    num_epchos =config["num_epochs"]
    dataset_type =config["dataset_type"]

    train,test= config_dataloaders(block_size,batch_size,target,dataset_type)
    model = create_model(hidden_units,output_shape,model_type,train,test)
    
    model.load_state_dict(torch.load(model_path))

    all_preds, all_trues, all_features,all_paths = evaluate_model(model,test,dataset_type)
    return all_preds,all_trues,all_features,all_paths


if __name__ == "__main__":
    
    train_and_predict(block_size=30,
                      batch_size=64,
                      target=0,
                      hidden_units=64,
                      output_shape=4,
                      model_type="LSTM_MIX",
                      learning_rate=0.001,
                      num_epchos=50,
                      dropout=0.2,
                      num_layers=2,
                      dataset_type="SequenceBlockWindowDataset")
    
    
    
   

