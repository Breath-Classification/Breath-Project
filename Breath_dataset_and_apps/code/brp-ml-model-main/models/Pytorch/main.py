#data_loader
from data_loader import create_dataloaders
from torchvision import transforms
#models
from GRUMODELPYTROCH import GruModel
from GRUMODELPYTROCH import GRUAttentionModel
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
import engine_without_epsilon
#libraries
import torch
import wandb 
from enum import Enum
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from HMM import viterbi_algorithm


#Constants
class SensorType(Enum):
    TENSOMETER = {"name": "tens", "size": 6}
    ACCELEROMETER = {"name": "acc", "size": 12}
    WIT_ACCELEROMETER = {"name": "acc", "size": 12}    
NUM_EPOCHS = 60
LEARNING_RATE = 0.01 #dla LSTM 0.001
BATCHES = 32
BLOCK_SIZE=[30]
SENSOR = SensorType.TENSOMETER
SENSOR_NAME = SENSOR.value["name"]
TARGET = 2

#Usage of Wandb 
use_wandb = input("Włączyć W&B? (y/n): ").strip().lower()

print (use_wandb)
if use_wandb != "y":
    os.environ["WANDB_DISABLED"] = "true"


def train_and_predict():
    for i, block in enumerate(BLOCK_SIZE):
        
        #logs
        wandb.init(
            mode="online" if  use_wandb=='y' else "disabled",
            project="GRU-optymalization",
            name=f"Transformer {block} ",
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
        model = create_model(block)
        train,test =create_train_test(block)
        
        
        #loss_fn = FocalLossAdaptive(gamma=2)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        engine.train(model, train, test, optimizer, loss_fn, NUM_EPOCHS, "cpu")

        wandb.finish()
    
    #saving model
    if use_wandb == 'n':
        filename = input("input name of the saved model")
        torch.save(model.state_dict(), f"saved_models/{filename}.pth")
    else:
        torch.save(model.state_dict(), f"saved_models/Class_Weightening_{block}.pth")
    all_preds, all_trues, all_features = evaluate_model(model,test)
    return all_preds, all_trues, all_features
    
   
def evaluate_model(model, test):
    all_preds = []
    all_trues = []

    all_features = []
    model.eval()
    return_sequence = True # USED FOR HMM
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
            return all_preds, all_trues, all_features
    else:
        print("Uzywam HMM")
        return
        with torch.no_grad():
            for X, y in test:
                # HMM all output not just last hidden state
                output = model(X, return_sequence=True)   # (batch, T, hidden)
                logits = model.fc(output)                 # (batch, T, 4)
                log_probs = torch.log_softmax(logits, -1)
                
                batch_paths = []
                for b in range(log_probs.size(0)):
                    preds_seq = log_probs[b]  
                    path = viterbi_algorithm(preds_seq)
                    batch_paths.append(torch.tensor(path))

                batch_paths = torch.stack(batch_paths) 
                    
                all_preds.append(batch_paths.cpu())
                all_trues.append(y.cpu())
                all_features.append(X.cpu())

            all_preds = torch.cat(all_preds, dim=0)
            all_trues = torch.cat(all_trues, dim=0)
            all_features = torch.cat(all_features, dim=0)

            print(all_trues)
            print(all_features.size())

            all_features = all_features.mean(dim=1)
            return all_preds, all_trues, all_features
        
    
def create_model(block):
    data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    train, test = create_dataloaders(transform=data_transform,
                                         batch_size=BATCHES,
                                         block_size=block,
                                         target=TARGET)
       
    X_batch, y_batch = next(iter(train))
        
        
    input_shape = X_batch.shape[2]
    hidden_units = 64
    output_shape = 4

    if use_wandb == 'n':
        print("Model:")
        print("1 - LSTM_ATTENTION")
        print("2 - LSTM_STACKED")
        print("3 - GruModel")
        print("4 - LSTM_BIDIRECTIONAL")
        print("5 - GRUAttentionModel")
        print("6 - LSTM_BASE")
        print("7 - LSTM_DROPOUT")
        print("8 - LSTM_CONV1")
        print("9 - Transformer")

        model_choice = input().strip()

        if model_choice == "1":
            model = LSTM_ATTENTION(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "2":
            model = LSTM_STACKED(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "3":
            model = GruModel(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "4":
            model = LSTM_BIDIRECTIONAL(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "5":
            model = GRUAttentionModel(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "6":
            model = LSTM_BASE(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "7":
            model = LSTM_DROPOUT(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "8":
            model = LSTM_CONV1(input_shape=input_shape,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        elif model_choice == "9":
            model = Transformer(input_shape=input_shape,d_model=32,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
        else:
            print("error")
    else:
        model = Transformer(input_shape=input_shape,d_model=32,
                                    hidden_units=hidden_units,
                                    output_shape=output_shape)
    return model
    
def create_train_test(block):
    data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    train, test = create_dataloaders(transform=data_transform,
                                         batch_size=BATCHES,
                                         block_size=block,
                                         target=TARGET)
    return train,test
def load_model_and_predict(model_path):
    
    model =create_model(block=30)
    train,test= create_train_test(block=30)
    model.load_state_dict(torch.load(model_path))
    all_preds, all_trues, all_features = evaluate_model(model,test)
    return all_preds,all_trues,all_features


if __name__ == "__main__":
    y_pred, y_true, X = train_and_predict()
    print("PRED:", y_pred)
    print("TRUE:", y_true)
   

