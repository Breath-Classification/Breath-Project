from data_loader import create_dataloaders
from torchvision import transforms
from GRUMODELPYTROCH import GruModel
from GRUMODELPYTROCH import Seq2SeqGRU
from GRUMODELPYTROCH import GRUAttentionModel
from LSTM.LSTM_Base import LSTM_BASE
from LSTM.LSTM_Dropout import LSTM_DROPOUT
from LSTM.LSTM_Stacked import LSTM_STACKED
from LSTM.LSTM_Bidirectional import LSTM_BIDIRECTIONAL
from LSTM.LSTM_Conv1 import LSTM_CONV1
from LSTM.LSTM_Attention import LSTM_ATTENTION
from Weightening.focal_loss import FocalLoss
from Weightening.adaptive_focal_loss import FocalLossAdaptive
import torch
import engine
import engine_without_epsilon
import Seq2SeqEngine
import wandb 
from enum import Enum
import torch.nn.functional as F
import matplotlib.pyplot as plt
class SensorType(Enum):
    TENSOMETER = {"name": "tens", "size": 6}
    ACCELEROMETER = {"name": "acc", "size": 12}
    WIT_ACCELEROMETER = {"name": "acc", "size": 12}
    
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
BATCHES = 32
BLOCK_SIZE=[30]
SENSOR = SensorType.TENSOMETER
SENSOR_NAME = SENSOR.value["name"]
TARGET = 2

def train_and_predict():
    for i, block in enumerate(BLOCK_SIZE):
        
        #logs
        wandb.init(
            project="GRU-optymalization",
            name=f"LSTM_ATTENTION 2 layers adaptive block ={block} ",
            group="test",
            config={
                "epochs": NUM_EPOCHS,
                "batch_size": BATCHES,
                "lr": LEARNING_RATE,
                "block_size":BLOCK_SIZE,
                "model":"GRU",
                "sensor":SENSOR_NAME,
                "loss_fn":"CrossEntropyLoss",
                "optimizer":"Adam"
            }
        )

        
        model = create_model(block)
        train,test =create_train_test(block)
        
        weights = torch.tensor([1.0, 1.0, 1.0, 2.25])

        #loss_fn = FocalLoss(gamma=2)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        engine.train(model, train, test, optimizer, loss_fn, NUM_EPOCHS, "cpu")

        wandb.finish()
        
    torch.save(model.state_dict(), f"saved_models/Class_Weightening_{block}.pth")
    all_preds, all_trues, all_features = evaluate_model(model,test)
    return all_preds, all_trues, all_features
    
   
def evaluate_model(model, test):
    all_preds = []
    all_trues = []

    all_features = []
    model.eval()
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

    model = LSTM_ATTENTION(input_shape=input_shape,
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
   

