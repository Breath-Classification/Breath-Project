
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
               device: torch.device,
               lambda_con0 : float,
               lambda_con1 : float,
               lambda_con2 : float,
               lambda_con3 : float
               ) -> Tuple[float, float]:

  model.train()
  train_loss, train_acc = 0, 0
  
  for batch, (X, y) in enumerate(dataloader):

      X, y = X.to(device), y.to(device)

      emissions = model(X)  # (B, seq_len, num_classes)
      mask = torch.ones_like(y, dtype=torch.bool)
      loss = -model.crf(emissions,y,mask)
      
      duration_penalty0 =min_duration_loss(emissions,0,2) 
      duration_penalty1 =min_duration_loss(emissions,1,2)
      duration_penalty2 =min_duration_loss(emissions,2,2)
      duration_penalty3 =min_duration_loss(emissions,3,2)
      
      #in_ex_penalty = 0 #inhale_exhale_correlation(emissions,0,10)
      
      
      #print("czy liczyony gradient")
      #print(duration_penalty0.requires_grad)
      #print(duration_penalty0.grad_fn)
      
      loss = loss.mean()  
      
      loss = loss + lambda_con0*duration_penalty0 +lambda_con1*duration_penalty1 +lambda_con2*duration_penalty2 +lambda_con3*duration_penalty3# +lambda_con0*in_ex_penalty


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
              device: torch.device) -> Tuple[float, float, float, float]:

    model.eval()
    test_loss = 0
    # Keep predictions in their original temporal order.  This is the same
    # ordering used by Visualisation/evaluation_metrics.py.
    all_predictions, all_targets = [], []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            emissions = model(X)  # (B, seq_len, num_classes)
            loss = -model.crf(emissions, y, reduction='mean')
            test_loss += loss.item()
            
            pred_seq = model.crf.decode(emissions)
            pred_tensor = torch.tensor(pred_seq, device=y.device)
            pred_tensor = pred_tensor.T 
            all_predictions.extend(pred_tensor.reshape(-1).cpu().tolist())
            all_targets.extend(y.reshape(-1).cpu().tolist())
    
    test_loss /= len(dataloader)

    correct = 0
    for index, (prediction, target) in enumerate(zip(all_predictions, all_targets)):
        if prediction == target or acceptable_error(
            all_predictions, all_targets, index, EPSILON
        ):
            correct += 1
    epsilon_accuracy = correct / len(all_targets) if all_targets else 0.0

    true_transitions = sum(
        previous != current
        for previous, current in zip(all_targets, all_targets[1:])
    )
    predicted_transitions = sum(
        previous != current
        for previous, current in zip(all_predictions, all_predictions[1:])
    )
    # Mirrors number_of_transitions_accuracy from
    # Visualisation/evaluation_metrics.py.  A value of 1.0 equals 100%.
    transition_accuracy = (
        true_transitions / predicted_transitions
        if predicted_transitions
        else 0.0
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

            if (
                target != prediction
                and not acceptable_error(
                    all_predictions, all_targets, cycle_index, EPSILON
                )
            ):
                is_cycle_accurate = False

            if target != 2:
                left_inhale = True

            if target == 2 and left_inhale:
                true_cycles += 1
                if is_cycle_accurate:
                    predicted_cycles += 1
                else:
                    # Equivalent to RR_error(cycle_predictions, cycle_targets, 1)
                    # from Visualisation/evaluation_metrics.py, kept local to
                    # avoid importing the visualisation module during training.
                    error_groups = 0
                    position = 0
                    while position < len(cycle_targets):
                        if cycle_predictions[position] == cycle_targets[position]:
                            position += 1
                            continue
                        # cycle_accuracy calls RR_error(cycle_pred, cycle_true, 1),
                        # so the second sequence supplies the error label.
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
          stop_point: float,
          lambda_con0 : float,
          lambda_con1 : float,
          lambda_con2 : float,
          lambda_con3 : float) -> Dict[str, List]:

  results = {
    "epoch":[],
    "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": [],
      "epsilon_accuracy": [],
      "transition_accuracy": [],
      "cycle_accuracy": [],
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
  end = False

  for epoch in tqdm(range(epochs)):
      #model.crf.transitions.data[1,3] = -1
      #model.crf.transitions.data[3,1] = -1
      
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          lambda_con0=lambda_con0,
                                          lambda_con1=lambda_con1,
                                          lambda_con2=lambda_con2,
                                          lambda_con3=lambda_con3)
      test_loss, epsilon_accuracy, transition_accuracy, cycle_accuracy = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      # Kept as an alias for backward-compatible LOSO result files.
      test_acc = epsilon_accuracy
      #print("Macierz przejść CRF:") 
      #print(model.crf.transitions)
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
