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

EPSILON = 2

def extract_transitions(sequence):
    transitions = []
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            transitions.append((i, sequence[i - 1], sequence[i]))
    return transitions

def match_transitions(predicted_transitions, true_transitions, epsilon=2):
    matched_true = set()
    tp = 0
    timing_errors = []

    for pred_position, pred_from, pred_to in predicted_transitions:
        best_match = None
        best_distance = None

        for true_index, (true_position, true_from, true_to) in enumerate(true_transitions):
            if true_index in matched_true:
                continue

            if pred_from != true_from or pred_to != true_to:
                continue

            distance = abs(pred_position - true_position)

            if distance <= epsilon:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match = true_index

        if best_match is not None:
            matched_true.add(best_match)
            tp += 1
            timing_errors.append(best_distance)

    fp = len(predicted_transitions) - tp
    fn = len(true_transitions) - tp

    return tp, fp, fn, timing_errors

def calculate_transition_metrics(all_predictions, all_targets, epsilon=2):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    timing_errors = []

    for predictions, targets in zip(all_predictions, all_targets):
        predicted_transitions = extract_transitions(predictions)
        true_transitions = extract_transitions(targets)

        tp, fp, fn, errors = match_transitions(
            predicted_transitions,
            true_transitions,
            epsilon
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        timing_errors.extend(errors)

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    transition_edtt_f1 = f1 * 100

    transition_timing_mae = (
        sum(timing_errors) / len(timing_errors)
        if timing_errors
        else 0.0
    )

    return transition_edtt_f1, transition_timing_mae

def train_step(model, dataloader, loss_fn, optimizer, device,
               lambda_con0, lambda_con1, lambda_con2, lambda_con3):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        emissions = model(X)
        mask = torch.ones_like(y, dtype=torch.bool)
        loss = -model.crf(emissions, y, mask)

        duration_penalty0 = min_duration_loss(emissions,0,2)
        duration_penalty1 = min_duration_loss(emissions,1,2)
        duration_penalty2 = min_duration_loss(emissions,2,2)
        duration_penalty3 = min_duration_loss(emissions,3,2)

        loss = loss.mean()
        loss = loss + lambda_con0*duration_penalty0 + lambda_con1*duration_penalty1 + lambda_con2*duration_penalty2 + lambda_con3*duration_penalty3

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_seq = model.crf.decode(emissions)
        pred_tensor = torch.tensor(pred_seq, device=y.device)
        pred_tensor = pred_tensor.T
        train_acc += (pred_tensor == y).float().mean().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device) -> Tuple[float, float, float, float]:
    model.eval()
    test_loss = 0
    all_predictions, all_targets = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            emissions = model(X)
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
        if prediction == target or acceptable_error(all_predictions, all_targets, index, EPSILON):
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

    transition_edtt_f1, transition_timing_mae = calculate_transition_metrics(
        [all_predictions],
        [all_targets],
        EPSILON
    )

    return (
        test_loss,
        epsilon_accuracy,
        transition_accuracy,
        cycle_accuracy,
        transition_edtt_f1,
        transition_timing_mae
    )

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn,
          epochs, device, stop, stop_point,
          lambda_con0, lambda_con1, lambda_con2, lambda_con3) -> Dict[str, List]:

    results = {
        "epoch":[],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epsilon_accuracy": [],
        "transition_accuracy": [],
        "cycle_accuracy": [],
        "transition_edtt_f1": [],
        "transition_timing_mae": [],
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
        train_loss, train_acc = train_step(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device,
            lambda_con0,
            lambda_con1,
            lambda_con2,
            lambda_con3
        )

        test_loss, epsilon_accuracy, transition_accuracy, cycle_accuracy, transition_edtt_f1, transition_timing_mae = test_step(
            model,
            test_dataloader,
            loss_fn,
            device
        )

        test_acc = epsilon_accuracy

        wandb.log({
            "epoch":epoch,
            "test_accuracy":test_acc,
            "epsilon_accuracy": epsilon_accuracy,
            "transition_accuracy": transition_accuracy,
            "cycle_accuracy": cycle_accuracy,
            "transition_edtt_f1": transition_edtt_f1,
            "transition_timing_mae": transition_timing_mae,
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
            f"transition_EDTT_F1: {transition_edtt_f1:.2f}% | "
            f"TT-MAE: {transition_timing_mae:.2f} | "
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
        results["transition_edtt_f1"].append(transition_edtt_f1)
        results["transition_timing_mae"].append(transition_timing_mae)
        results["max_test_acc"].append(maksimum_test_acc)

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