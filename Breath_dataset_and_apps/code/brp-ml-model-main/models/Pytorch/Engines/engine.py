
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
              device: torch.device) -> Tuple[float, float, float, float]:

  model.eval()
  test_loss = 0
  all_predictions, all_targets = [], []
  with torch.inference_mode():
      for X, y in dataloader:
          X, y = X.to(device), y.to(device)
          test_pred_logits = model(X)
          test_pred_labels = test_pred_logits.argmax(dim=1)
          batch_loss = 0.0

          for i in range(len(y)):
              if (
                  test_pred_labels[i] != y[i]
                  and not acceptable_error(test_pred_labels, y, i, EPSILON)
              ):
                  loss = loss_fn(
                      test_pred_logits[i].unsqueeze(0), y[i].unsqueeze(0)
                  )
                  batch_loss += loss.item()

          test_loss += batch_loss
          all_predictions.extend(test_pred_labels.cpu().tolist())
          all_targets.extend(y.cpu().tolist())

  test_loss = test_loss / len(dataloader)

  correct = sum(
      prediction == target or acceptable_error(
          all_predictions, all_targets, index, EPSILON
      )
      for index, (prediction, target) in enumerate(
          zip(all_predictions, all_targets)
      )
  )
  epsilon_accuracy = correct / len(all_targets) if all_targets else 0.0

  true_transitions = sum(
      previous != current
      for previous, current in zip(all_targets, all_targets[1:])
  )
  predicted_transitions = sum(
      previous != current
      for previous, current in zip(all_predictions, all_predictions[1:])
  )
  transition_accuracy = (
      true_transitions / predicted_transitions if predicted_transitions else 0.0
  )

  predicted_cycles = 0
  true_cycles = 0
  index = 0
  while index < len(all_targets):
      if all_targets[index] != 2:
          index += 1
          continue

      cycle_targets = []
      cycle_predictions = []
      is_cycle_accurate = True
      left_inhale = False
      for cycle_index in range(index + 1, len(all_targets)):
          target = all_targets[cycle_index]
          prediction = all_predictions[cycle_index]
          cycle_targets.append(target)
          cycle_predictions.append(prediction)

          if target != prediction and not acceptable_error(
              all_predictions, all_targets, cycle_index, EPSILON
          ):
              is_cycle_accurate = False
          if target != 2:
              left_inhale = True

          if target == 2 and left_inhale:
              true_cycles += 1
              if is_cycle_accurate:
                  predicted_cycles += 1
              else:
                  error_groups = 0
                  position = 0
                  while position < len(cycle_targets):
                      if cycle_predictions[position] == cycle_targets[position]:
                          position += 1
                          continue
                      error_label = cycle_targets[position]
                      error_groups += 1
                      position += 1
                      while (
                          position < len(cycle_targets)
                          and cycle_predictions[position] != cycle_targets[position]
                          and cycle_targets[position] == error_label
                      ):
                          position += 1
                  if error_groups <= 1:
                      predicted_cycles += 1
              index = cycle_index + 1
              break
      else:
          index = len(all_targets)

  cycle_accuracy = predicted_cycles / true_cycles if true_cycles else 0.0
  return test_loss, epsilon_accuracy, transition_accuracy, cycle_accuracy

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
      "epsilon_accuracy": [],
      "transition_accuracy": [],
      "cycle_accuracy": [],
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
      test_loss, epsilon_accuracy, transition_accuracy, cycle_accuracy = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      # test_acc remains an alias for compatibility with existing result files.
      test_acc = epsilon_accuracy
      
      #Save the results to the results dictionary and to wandb

      wandb.log({
          "epoch":epoch,
          "test_accuracy":test_acc,
          "epsilon_accuracy": epsilon_accuracy,
          "transition_accuracy": transition_accuracy,
          "cycle_accuracy": cycle_accuracy,
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
          f"epsilon_acc: {epsilon_accuracy:.4f} | "
          f"transition_acc: {transition_accuracy:.4f} | "
          f"cycle_acc: {cycle_accuracy:.4f} | "
          f"max_test_acc: {maksimum_test_acc:.4f} | "
      )
    
          
      
      results["epoch"].append(epoch+1)
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
      results["epsilon_accuracy"].append(epsilon_accuracy)
      results["transition_accuracy"].append(transition_accuracy)
      results["cycle_accuracy"].append(cycle_accuracy)
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
