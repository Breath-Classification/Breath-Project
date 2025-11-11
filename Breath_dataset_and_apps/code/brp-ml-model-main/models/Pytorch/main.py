from data_loader import create_dataloaders
from torchvision import transforms
from GRUMODELPYTROCH import GruModel
import torch
import engine
import wandb 
from enum import Enum

class SensorType(Enum):
    TENSOMETER = {"name": "tens", "size": 6}
    ACCELEROMETER = {"name": "acc", "size": 12}
    WIT_ACCELEROMETER = {"name": "acc", "size": 12}


if __name__ == "__main__":

    NUM_EPOCHS = 64
    LEARNING_RATE = 0.001
    BATCHES = 32
    BLOCK_SIZE=[14]
    SENSOR = SensorType.TENSOMETER
    SENSOR_NAME = SENSOR.value["name"]
    
    for i, block in enumerate(BLOCK_SIZE):
        wandb.init(  #dane konkretnej proby
                project="GRU-optymalization",
                name=f" scheduler progressive Block_{block}",
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

        train,test = create_dataloaders(transform=data_transform,batch_size=BATCHES, block_size=block)

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


