#data_loader
from data_loader import create_dataloaders
from torchvision import transforms
#models
from GRU.GruModel import GruModel
from GRU.GruAttention import GRUAttentionModel
from GRU.GruSeq import Seq2SeqGRU
from LSTM.LSTM_Base import LSTM_BASE
from LSTM.LSTM_Dropout import LSTM_DROPOUT
from LSTM.LSTM_Stacked import LSTM_STACKED
from LSTM.LSTM_Bidirectional import LSTM_BIDIRECTIONAL
from LSTM.LSTM_Conv1 import LSTM_CONV1
from LSTM.LSTM_Attention import LSTM_ATTENTION
from Transformers.transformer import Transformer
#loss functions
from Weightening.focal_loss import FocalLoss
from Weightening.adaptive_focal_loss import FocalLossAdaptive
#engines
import engine
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
                      loos_type ="CrossEntropyLoss", optimizer_type="Adam",
                      dropout =0, num_layers=2, dim_feedforward =64, 
                      nhead  =2, d_model=32):
   
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
    train,test =config_dataloaders(block_size,batch_size,target)

    model = create_model(hidden_units,output_shape,model_type,train,test,dropout,num_layers,dim_feedforward,nhead, d_model)
    loss_fn = create_loos_function(loos_type)
    optimizer = create_optimizer(optimizer_type,model,learning_rate)

    results =engine.train(model, train, test, optimizer, loss_fn, num_epchos, "cpu", True, 0.931) #stop i set 

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
        "optimizer_type":optimizer_type
    }
    
    save_model(model,config)
    save_model_mobile(model)

    all_preds, all_trues, all_features,_ = evaluate_model(model,test)
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
    os.makedirs("mobile_models", exist_ok=True)
    traced_script_module.save(f"mobile_models/{filename}.pt")
    print(f"TorchScript model saved as mobile_models/{filename}.pt")
    
def save_model(model, config):
    #saving model
    if use_wandb == 'n':
        filename = input("input name of the saved model ")
        torch.save(model.state_dict(), f"saved_models/{filename}.pth")
    else:
        torch.save(model.state_dict(), f"saved_models/Class_Weightening_{block_size}.pth")
    
    with open(f"saved_models/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def evaluate_model(model, test):
    all_preds = []
    all_trues = []

    all_features = []
    model.eval()
    return_sequence = False # USED FOR HMM
    if(not return_sequence):
        print("hej")
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
            print(all_trues)
            print(all_features.size())
            all_features = all_features.mean(dim=1) 
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


def config_dataloaders(block_size,batch_size,target):
    data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    train, test = create_dataloaders(transform=data_transform,
                                         batch_size=batch_size,
                                         block_size=block_size,
                                         target=target)

    return train,test

def create_model(hidden_units,output_shape,model_type,train,test, dropout=0, num_layers=2, dim_feedforward =64, nhead  =2, d_model=32):
    
    X_batch, y_batch = next(iter(train))
    input_shape = X_batch.shape[2]

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

    train,test= config_dataloaders(block_size,batch_size,target)
    model = create_model(hidden_units,output_shape,model_type,train,test)
    
    model.load_state_dict(torch.load(model_path))

    all_preds, all_trues, all_features,all_paths = evaluate_model(model,test)
    return all_preds,all_trues,all_features,all_paths


if __name__ == "__main__":
    
    train_and_predict(30,32,0,64,4,"LSTM_BASE",0.001,40)
    
    
    
   

