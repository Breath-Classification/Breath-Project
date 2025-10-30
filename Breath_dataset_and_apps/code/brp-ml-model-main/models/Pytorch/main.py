from data_loader import create_dataloaders
from torchvision import transforms
from GRUMODELPYTROCH import GruModel
import torch
import engine


NUM_EPOCHS = 100
LEARNING_RATE = 0.001

BATCHES = 32
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

train,test = create_dataloaders(transform=data_transform,batch_size=BATCHES)

X_batch, y_batch = next(iter(train))
print("Batch shapes:", X_batch.shape, y_batch.shape)
input_shape = X_batch.shape[2]  # liczba cech (2)
hidden_units = 64
output_shape = 4

model = GruModel(input_shape=input_shape,hidden_units=hidden_units,output_shape=output_shape)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

engine.train(model,train,test,optimizer,loss_fn,NUM_EPOCHS,"cpu")

with torch.no_grad():
    outputs=model(X_batch)
    print('dziala test')
    print(outputs.shape)
    preds = torch.argmax(outputs, dim=1)  # wybieramy indeks największego logitu
    print("Predykcje:", preds)            # tensor z 32 przewidywaniami

    # Opcjonalnie porównanie z prawdziwymi etykietami
    print("Prawdziwe etykiety:", y_batch)



i=-1
for X_batch, y_batch in train:
    if(i<=0):
        print("🔹 X_batch shape:", X_batch.shape)
        print("🔹 y_batch shape:", y_batch.shape)
        print("X_batch example:", X_batch[0])
        print("y_batch example:", y_batch[0])
        i=i+1
    else:
        break  # tylko pierwszy batch, żeby nie wypisywać wszystkiego