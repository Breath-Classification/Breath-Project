from data_loader import create_dataloaders
from torchvision import transforms
from GRUMODELPYTROCH import GruModel
from GRUMODELPYTROCH import Seq2SeqGRU
from GRUMODELPYTROCH import GRUAttentionModel
import torch
import engine
import Seq2SeqEngine
import wandb 
from enum import Enum
import torch.nn.functional as F
import matplotlib.pyplot as plt
class SensorType(Enum):
    TENSOMETER = {"name": "tens", "size": 6}
    ACCELEROMETER = {"name": "acc", "size": 12}
    WIT_ACCELEROMETER = {"name": "acc", "size": 12}

def train_and_predict():
    NUM_EPOCHS = 128
    LEARNING_RATE = 0.001
    BATCHES = 32
    BLOCK_SIZE=[30,31,32,33,34,35,80]
    SENSOR = SensorType.TENSOMETER
    SENSOR_NAME = SENSOR.value["name"]
    TARGET = 2

    model = None
    last_X_batch = None

    for i, block in enumerate(BLOCK_SIZE):
        wandb.init(
            project="GRU-optymalization",
            name=f"more epochs shuffle = false block ={block}",
            group="Shuffle",
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

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        engine.train(model, train, test, optimizer, loss_fn, NUM_EPOCHS, "cpu")

        wandb.finish()

    all_preds = []
    all_trues = []

    all_features = []
    model.eval()
    with torch.no_grad():
        for X, y in test:
            y_pred = model(X)  # (batch, target_size, num_classes) jeśli seq2seq
            pred_classes = torch.argmax(y_pred, dim=1)  # (batch, target_size) 
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
    '''
     with torch.no_grad():
        for X, y in test:
            y_pred = model(X)  # (batch, target_size, num_classes) jeśli seq2seq
            pred_classes = torch.argmax(y_pred, dim=1)  # (batch, target_size) 
            all_preds.append(pred_classes.cpu())
            all_trues.append(y.cpu())
            all_features.append(X.cpu())
        all_preds =torch.cat(all_preds)
        all_trues =torch.cat(all_trues)
        all_features =torch.cat(all_features)
        all_features = all_features.mean(dim=1) 
        return all_preds, all_trues, all_features
    '''




if __name__ == "__main__":
    y_pred, y_true, X = train_and_predict()
    print("PRED:", y_pred)
    print("TRUE:", y_true)
   

