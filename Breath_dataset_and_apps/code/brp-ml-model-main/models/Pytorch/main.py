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


if __name__ == "__main__":

    NUM_EPOCHS = 64
    LEARNING_RATE = 0.001
    BATCHES = 32
    BLOCK_SIZE=[12]
    SENSOR = SensorType.TENSOMETER
    SENSOR_NAME = SENSOR.value["name"]
    TARGET =2
    
    for i, block in enumerate(BLOCK_SIZE):
        wandb.init(  #dane konkretnej proby
                project="GRU-optymalization",
                name=f"Attention Dropout block={block}",
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
        print("wandb.run after init:", wandb.run)


        data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
        ])

        train,test = create_dataloaders(transform=data_transform,batch_size=BATCHES, block_size=block, target=TARGET)

        X_batch, y_batch = next(iter(train))
        input_shape = X_batch.shape[2]  # liczba cech (2)
        hidden_units = 64
        output_shape = 4 #liczba kategorii

        #model = GruModel(input_shape=input_shape,hidden_units=hidden_units,output_shape=output_shape)
        model = GRUAttentionModel(input_shape=input_shape,hidden_units=hidden_units,output_shape=output_shape)
        loss_fn = torch.nn.CrossEntropyLoss() #loss function
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=LEARNING_RATE)

        engine.train(model,train,test,optimizer,loss_fn,NUM_EPOCHS,"cpu")
        
        wandb.finish()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_batch)  # (batch, seq_len, output_dim)
        target_size = y_batch.shape[1]
        y_pred = y_pred[:, -target_size:, :]  # wybieramy ostatnie target_size kroków

    # Wybieramy np. pierwszą próbkę z batcha
    for i in range(32):
        y_true_sample = y_batch[i].numpy()
        y_pred_sample = y_pred[i].numpy()  # (target_size, output_dim) jeśli output_dim=1

        # Jeśli output_dim=1 → spłaszczamy
        
        y_pred_sample = torch.tensor(y_pred_sample)
        probs = F.softmax(y_pred_sample, dim=1)
        pred_classes = torch.argmax(probs, dim=1)
        print("probka ",i)
        print("prawidziwe")
        print(y_true_sample)
        print("przewidywania")
        print(pred_classes)
    """
    wandb.init(  #dane konkretnej proby
                project="GRU-optymalization",
                name="data load Sequence",
                config={
                    "epochs": NUM_EPOCHS,
                    "batch_size": BATCHES,
                    "lr": LEARNING_RATE,
                    "model":"GRU",
                    "sensor":SENSOR_NAME,
                    "loss_fn":"CrossEntropyLoss",
                    "optimizer":"Adam"
                }
            )
    print("wandb.run after init:", wandb.run)


    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
        ])

    train,test = create_dataloaders(transform=data_transform,batch_size=BATCHES, block_size=0)

    X_batch, y_batch = next(iter(train))
    input_shape = X_batch.shape[2]  # liczba cech (2)
    hidden_units = 64
    output_shape = 4 #liczba kategorii

    model = GruModel(input_shape=input_shape,hidden_units=hidden_units,output_shape=output_shape)

    loss_fn = torch.nn.CrossEntropyLoss() #loss function
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=LEARNING_RATE)

    engine.train(model,train,test,optimizer,loss_fn,NUM_EPOCHS,"cpu")
    wandb.finish()    
    """


