
"""
Contains functions for training and testing a PyTorch model.  https://github.com/mrdbourke/pytorch-deep-learning
Code is adapted for the dataset, extended with an error tolerance mechanism, 
returns accuracy and other parametres.
"""

import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import wandb
from scripts.error_tolerance import acceptable_error
from scripts.HMM import viterbi_algorithm



EPSILON = 2 #Constant that defines when an error can be considered acceptable 
            #(in this case, up to 2 readings forward or backward)

def train_step(model: torch.nn.Module,  #Step where the model trains without considering error correction
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


    
def test_step(model: torch.nn.Module,   # Step where the model's performance is evaluated on the test data taking error correction into account
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

  model.eval() 
  test_loss, test_acc = 0, 0
  correct = 0
  total = 0
  with torch.inference_mode():
      
      for batch, (X, y) in enumerate(dataloader):
          all_paths = []
          X, y = X.to(device), y.to(device)
  
          test_pred_logits = model(X)
          test_pred_labels = test_pred_logits.argmax(dim=1)
          batch_loss = 0.0
          '''
          output = model(X)   # (batch, T, hidden)
          logits = model.fc(output)                 # (batch, T, 4)
          log_probs = torch.log_softmax(logits, -1)
          '''
                
                
          '''
          for i in range(32):
              path = viterbi_algorithm(log_probs[i])
              last_label = torch.tensor(path[-1]) 
              all_paths.append(last_label.cpu())
          
          all_paths = torch.tensor(all_paths)
          test_pred_labels = all_paths.clone()     
          '''
          #Error tolerance
          #When the prediction error occurs at the boundary of two classes 
          #and is within a distance of at most EPSILON, 
          #we do not count this prediction as incorrect
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
        
  test_loss = test_loss / len(dataloader)
  test_acc = correct / total 
  return test_loss, test_acc

def train(model: torch.nn.Module,                      #Main training loop
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          stop: bool,
          stop_point: float) -> Dict[str, List]:

  results = {
    "epoch":[],
    "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": [],
      "max_test_acc":[],
  }
  end = False
  best_scores={
    "max_test_acc":0.0,
    "max_train_acc":0.0,
    "final_test_loss":0.0,
    "final_train_loss":0.0
  }
  
  maksimum_test_acc =0
  maksimum_train_acc =0

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
      
      #Save the results to the results dictionary and to wandb

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
      
      #Save the model when its performance exceeds the stop_point
      if(stop==True and test_acc>=stop_point):
        end = True
        break

  wandb.summary.update({
    "max_test_acc":best_scores["max_test_acc"],
    "max_train_acc":best_scores["max_train_acc"],
    "final_test_loss":best_scores["final_test_loss"],
    "final_train_loss":best_scores["final_train_loss"]
  })
  
  return results,end