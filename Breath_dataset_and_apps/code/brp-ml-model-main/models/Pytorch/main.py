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
    NUM_EPOCHS = 64
    LEARNING_RATE = 0.001
    BATCHES = 32
    BLOCK_SIZE=[6,7,8,9,10,11]
    SENSOR = SensorType.TENSOMETER
    SENSOR_NAME = SENSOR.value["name"]
    TARGET = 2

    model = None
    last_X_batch = None

    for i, block in enumerate(BLOCK_SIZE):
        wandb.init(
            project="GRU-optymalization",
            name=f"Attention Dropout 2 after 0.03 and 0.07 1 block={block}",
            group="Attention",
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

        model = GRUAttentionModel(input_shape=input_shape,
                                  hidden_units=hidden_units,
                                  output_shape=output_shape)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        engine.train(model, train, test, optimizer, loss_fn, NUM_EPOCHS, "cpu")

        last_X_batch = X_batch    # przechowujemy batch
        last_y_batch = y_batch

        wandb.finish()

    # ---- PREDYKCJE ----
    model.eval()
    with torch.no_grad():
        y_pred = model(last_X_batch)  # shape: (batch, seq_len, output_dim)
        target_size = last_y_batch.shape[1]
        y_pred = y_pred[:, -target_size:, :]  # wybieramy końcówkę sekwencji

    # predykcje klas
    y_pred_classes = torch.argmax(torch.softmax(y_pred, dim=2), dim=2)

    return y_pred_classes, last_y_batch


if __name__ == "__main__":
    y_pred, y_true = train_and_predict()
    print("PRED:", y_pred)
    print("TRUE:", y_true)
   

