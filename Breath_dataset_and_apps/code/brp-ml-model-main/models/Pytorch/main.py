from data_loader import create_dataloaders
from torchvision import transforms
from GRUMODELPYTROCH import GruModel
from GRUMODELPYTROCH import Seq2SeqGRU
from GRUMODELPYTROCH import GRUAttentionModel
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
    
NUM_EPOCHS = 30
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
            name=f"more epochs shuffle = false block ={block}",
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
        
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        engine.train(model, train, test, optimizer, loss_fn, NUM_EPOCHS, "cpu")

        wandb.finish()
        
    torch.save(model.state_dict(), "saved_models/model.pth")
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

    model = GruModel(input_shape=input_shape,
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
    all_preds, all_trues, all_features = evaluate_model(model,train)
    return all_preds,all_trues,all_features


if __name__ == "__main__":
    y_pred, y_true, X = train_and_predict()
    print("PRED:", y_pred)
    print("TRUE:", y_true)
   

