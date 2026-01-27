
"""
Contains functions for training and testing a PyTorch model.  https://github.com/mrdbourke/pytorch-deep-learning
"""
from HMM import viterbi_algorithm
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import wandb
import os
import sys
from scripts.error_tolerance import acceptable_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print("wandb.run before training:", wandb.run)

EPSILON = 2
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

      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


    
def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  model.eval() 
  

  test_loss, test_acc = 0, 0
  
  correct = 0
  total = 0
  return_sequence = True # USED FOR HMM
  if(not return_sequence):
    with torch.inference_mode():
        
        for batch, (X, y) in enumerate(dataloader):
          
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            test_pred_labels = test_pred_logits.argmax(dim=1)
            batch_loss = 0.0
            for i in range(len(y)):
              if(test_pred_labels[i]!=y[i]):
                if(acceptable_error(test_pred_labels,y,i, EPSILON)):
                  pass
                else:
                  loss = loss_fn(test_pred_logits[i].unsqueeze(0), y[i].unsqueeze(0))
                  batch_loss += loss.item()
                  
              if test_pred_labels[i] == y[i]:
                  correct += 1
              else:
                  if acceptable_error(test_pred_labels, y, i,EPSILON):
                      correct += 1  
              total += 1
            
          
            test_loss += batch_loss
            #test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            
    test_loss = test_loss / len(dataloader)
    test_acc = correct / total 
    return test_loss, test_acc
  else:
     with torch.inference_mode():
        
        for batch, (X, y) in enumerate(dataloader):
          
            X, y = X.to(device), y.to(device)
            output = model(X, return_sequence=True)   # (batch, T, hidden)
            logits = model.fc(output)                 # (batch, T, num_classes)
            log_probs = torch.log_softmax(logits, dim=-1)
            batch_loss = 0.0
            for b in range(log_probs.size(0)):
              preds_seq = log_probs[b]              # (T, num_classes)
              true_seq  = y[b]                      # (T,)

              path = viterbi_algorithm(preds_seq)   # list[T]
              path = torch.tensor(path, device=device)

              for i in range(len(true_seq)):

                  if path[i] != true_seq[i]:
                      if acceptable_error(path, true_seq, i, EPSILON):
                          pass
                      else:
                          loss = loss_fn(
                              logits[b, i].unsqueeze(0),
                              true_seq[i].unsqueeze(0)
                          )
                          batch_loss += loss.item()

                  if path[i] == true_seq[i]:
                      correct += 1
                  else:
                      if acceptable_error(path, true_seq, i, EPSILON):
                          correct += 1

                  total += 1

            test_loss += batch_loss
              
            
  test_loss = test_loss / len(dataloader)
  test_acc = correct / total 
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