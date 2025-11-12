
"""
Contains functions for training and testing a PyTorch model.  https://github.com/mrdbourke/pytorch-deep-learning
"""
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import wandb
print("wandb.run before training:", wandb.run)
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

  model.train()
  
  train_loss, train_acc = 0, 0
  
  for batch, (X, y) in enumerate(dataloader):

      X, y = X.to(device), y.to(device)

      y_pred = model(X)
      
      target_size = y.shape[1]
      y_pred = y_pred[:, -target_size:, :] # nasze Y to teraz wektor przyszlych cech

      loss = loss_fn(y_pred.reshape(-1, y_pred.shape[2]), y.reshape(-1))
      train_loss += loss.item() 

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      y_pred_class = torch.argmax(y_pred, dim=2)  # (batch, target_size)
      train_acc += (y_pred_class == y).float().mean().item() 

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  model.eval() 
  

  test_loss, test_acc = 0, 0
  

  with torch.inference_mode():
     
      for batch, (X, y) in enumerate(dataloader):
         
          X, y = X.to(device), y.to(device)
  
          
          test_pred_logits = model(X)
          
          target_size = y.shape[1]
          test_pred_logits = test_pred_logits[:, -target_size:, :] 

      
          loss = loss_fn(test_pred_logits.reshape(-1, test_pred_logits.shape[2]), y.reshape(-1))
          test_loss += loss.item()
          
         
          test_pred_labels = torch.argmax(test_pred_logits, dim=2)  # (batch, target_size)
          test_acc += (test_pred_labels == y).float().mean().item() 
          
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

  results = {
    "epoch":[],
    "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": [],
      "max_test_acc":[]
  }
  best_scores={
    "max_test_acc":0.0,
    "max_train_acc":0.0,
    "final_test_loss":0.0,
    "final_train_loss":0.0
  }
  
  maksimum_test_acc =0
  maksimum_train_acc =0
  #scheduler = StepLR(optimizer, step_size=epochs//2, gamma=0.5)
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      #scheduler.step(test_loss)
 
      wandb.log({
          "epoch":epoch,
          "test_accuracy":test_acc,
          "train_accuracy":train_acc,
          "test_loss":test_loss,
          "train_loss":train_loss
        })
   
      
      if(maksimum_test_acc<test_acc):
          maksimum_test_acc=test_acc
      if(maksimum_train_acc<train_acc):
          maksimum_train_acc=train_acc
      
      best_scores["max_test_acc"]=maksimum_test_acc
      best_scores["max_train_acc"]=maksimum_train_acc
      if (epoch == epochs - 1):
        best_scores["final_test_loss"]=test_loss
        best_scores["final_train_loss"]=train_loss
          
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} | "
          f"max_test_acc: {maksimum_test_acc:.4f} | "
      )
    
          
      
      results["epoch"].append(epoch+1)
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
      results["max_test_acc"].append(maksimum_test_acc)

  wandb.summary.update({
    "max_test_acc":best_scores["max_test_acc"],
    "max_train_acc":best_scores["max_train_acc"],
    "final_test_loss":best_scores["final_test_loss"],
    "final_train_loss":best_scores["final_train_loss"]
  })
  return results