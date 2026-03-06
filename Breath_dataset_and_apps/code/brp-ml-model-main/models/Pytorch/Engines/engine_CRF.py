
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
from PhysicalConstraints.avg_class_lenght_loss import min_duration_loss
from PhysicalConstraints.breath_duration_loss import inhale_exhale_correlation



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

      emissions = model(X)  # (B, seq_len, num_classes)
      mask = torch.ones_like(y, dtype=torch.bool)
      loss = -model.crf(emissions,y,mask)
      
      duration_penalty0 =min_duration_loss(emissions,0,3)
      duration_penalty1 =min_duration_loss(emissions,1,3)
      duration_penalty2 =min_duration_loss(emissions,2,3)
      duration_penalty3 =min_duration_loss(emissions,3,3)
      
      in_ex_penalty =inhale_exhale_correlation(emissions,0,10)
      
      
      print("czy liczyony gradient")
      print(duration_penalty0.requires_grad)
      print(duration_penalty0.grad_fn)
      
      loss = loss.mean()  
      lambda_con0 =0.1
      loss = loss + lambda_con0*duration_penalty0 +lambda_con0*duration_penalty1 +lambda_con0*duration_penalty2 +lambda_con0*duration_penalty3+lambda_con0*in_ex_penalty


      train_loss += loss.item()

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()
      
      pred_seq = model.crf.decode(emissions)  # list[list]
      
      
      
      
      pred_tensor = torch.tensor(pred_seq, device=y.device)
      pred_tensor = pred_tensor.T # aby naprawic wymiray
      #print("y shape", y.shape)
      #print(pred_tensor.shape)
      train_acc += (pred_tensor == y).float().mean().item()

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


    
def test_step(model: torch.nn.Module,   # Step where the model's performance is evaluated on the test data taking error correction into account
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    model.eval()
    test_loss, correct, total = 0, 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            emissions = model(X)  # (B, seq_len, num_classes)
            loss = -model.crf(emissions, y, reduction='mean')
            test_loss += loss.item()
            
            pred_seq = model.crf.decode(emissions)
            pred_tensor = torch.tensor(pred_seq, device=y.device)
            pred_tensor = pred_tensor.T 
            # accuracy z tolerancją EPSILON
            for i in range(pred_tensor.shape[0]):
                for t in range(pred_tensor.shape[1]):
                    if pred_tensor[i, t] == y[i, t]:
                        correct += 1
                    elif acceptable_error(pred_tensor[i], y[i], t, EPSILON):
                        correct += 1
                    total += 1
    
    test_loss /= len(dataloader)
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

  for epoch in tqdm(range(epochs)):
      model.crf.transitions.data[1,3] = -1
      model.crf.transitions.data[3,1] = -1
      
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      print("Macierz przejść CRF:") 
      print(model.crf.transitions)
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
        break

  wandb.summary.update({
    "max_test_acc":best_scores["max_test_acc"],
    "max_train_acc":best_scores["max_train_acc"],
    "final_test_loss":best_scores["final_test_loss"],
    "final_train_loss":best_scores["final_train_loss"]
  })
  
  return results