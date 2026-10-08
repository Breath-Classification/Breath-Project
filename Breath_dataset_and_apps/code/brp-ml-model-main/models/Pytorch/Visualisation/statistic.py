import torch
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error
from scripts.error_tolerance import RR_error
def epsilon_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    epsilons = 10
    accuracy = [] 
    for epsilon in range(epsilons):
        correct_prediction=0
        for i in range(n_samples):
            if y_pred[i]==y_true[i]:
                correct_prediction+=1
            elif acceptable_error(y_pred,y_true,i,epsilon): 
                correct_prediction+=1
        accuracy.append((correct_prediction/n_samples) *100)    
    return accuracy

def epsilon_RR_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    accuracy= []
    epsilons =10
    for epsilon in range(epsilons):
        i=0
        pred_cycle=0
        true_cycle=0
        while i < n_samples:
            cycle_true = []
            cycle_pred = []
            if y_true[i] == 2:
                is_cycle = True
                start_inhale = True
                for j in range(i+1,n_samples):
                    cycle_true.append(y_true[j])
                    cycle_pred.append(y_pred[j])
                    if y_true[j] != y_pred[j]:
                        if not acceptable_error(y_pred,y_true,j,2):
                            is_cycle = False
                    if y_true[j]!=2:
                        start_inhale = False
                    if y_true[j]==2 and start_inhale==False and is_cycle==True:
                        pred_cycle+=1
                    elif y_true[j]==2 and start_inhale==False:
                        if RR_error(cycle_pred,cycle_true,epsilon):
                            pred_cycle+=1
                    if y_true[j]==2 and start_inhale==False:
                        true_cycle+=1
                        i=j+1
                        break
                    if j==n_samples-1:
                        i=n_samples+1     
            else:
                i+=1
        accuracy.append((pred_cycle/true_cycle) *100)
    return accuracy


def esilon_RR_accuracy(model_path):
    return epsilon_RR_accuracy(model_path)

def precision(model_path, actual_class):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    TruePositive = 0
    FalsePositive = 0
    for i in range(n_samples):
        if y_true[i] == y_pred[i] and actual_class == y_true[i]:
            TruePositive += 1
        if y_pred[i] == actual_class and y_true[i] != actual_class:
            FalsePositive += 1
            
    if TruePositive+FalsePositive == 0:
        return 0
    Precision = TruePositive / (TruePositive + FalsePositive)    
    return Precision

def recall(model_path, actual_class):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    TruePositive = 0
    FalseNegative = 0
    for i in range(n_samples):
        if y_true[i] == y_pred[i] and actual_class == y_true[i]:
            TruePositive += 1
        if y_pred[i] != actual_class and y_true[i] == actual_class:
            FalseNegative += 1
            
    if TruePositive+FalseNegative == 0:
        return 0
    Recall = TruePositive / (TruePositive + FalseNegative)    
    return Recall
    

def f_scale(precision, recall):
    F1_score = 2* precision*recall /(precision+recall)
    return F1_score


def extract_transitions(sequence):
 
    transitions = []

    for i in range(1, len(sequence)):
        previous_phase = sequence[i - 1]
        current_phase = sequence[i]

        if previous_phase != current_phase:
            transitions.append(
                (i, previous_phase, current_phase)
            )

    return transitions


def match_transitions(predicted_transitions, true_transitions, epsilon):
 
    matched_true = set()
    matched_pred = set()

    for pred_idx, (pred_position, pred_from, pred_to) in enumerate(
        predicted_transitions
    ):
        best_true_idx = None
        best_distance = None

        for true_idx, (true_position, true_from, true_to) in enumerate(
            true_transitions
        ):
            if true_idx in matched_true:
                continue

            if pred_from != true_from or pred_to != true_to:
                continue

            distance = abs(pred_position - true_position)

            if distance <= epsilon:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_true_idx = true_idx

        if best_true_idx is not None:
            matched_pred.add(pred_idx)
            matched_true.add(best_true_idx)

    TP = len(matched_pred)
    FP = len(predicted_transitions) - TP
    FN = len(true_transitions) - TP

    return TP, FP, FN


def transition_edtt_f1(model_path):

    y_pred, y_true, _, _ = load_model_and_predict(model_path)

    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().tolist()

    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().tolist()

    predicted_transitions = extract_transitions(y_pred)
    true_transitions = extract_transitions(y_true)

    epsilon = 2

    TP, FP, FN = match_transitions(
        predicted_transitions,
        true_transitions,
        epsilon
    )

    if TP + FP == 0:
        precision_value = 0
    else:
        precision_value = TP / (TP + FP)

    if TP + FN == 0:
        recall_value = 0
    else:
        recall_value = TP / (TP + FN)

    if precision_value + recall_value == 0:
        f1 = 0
    else:
        f1 = (2 * precision_value * recall_value   / (precision_value + recall_value))

    return f1 * 100

def transition_timing_mae(model_path):

    y_pred, y_true, _, _ = load_model_and_predict(model_path)

    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().tolist()

    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().tolist()

    predicted_transitions = extract_transitions(y_pred)
    true_transitions = extract_transitions(y_true)

    epsilon = 2

    matched_true = set()
    transition_errors = []

    for pred_idx, (
        pred_position,
        pred_from,
        pred_to
    ) in enumerate(predicted_transitions):

        best_true_idx = None
        best_distance = None

        for true_idx, (
            true_position,
            true_from,
            true_to
        ) in enumerate(true_transitions):

            if true_idx in matched_true:
                continue

            if pred_from != true_from or pred_to != true_to:
                continue

            distance = abs(pred_position - true_position)

            if distance <= epsilon:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_true_idx = true_idx

        if best_true_idx is not None:
            matched_true.add(best_true_idx)
            transition_errors.append(best_distance)

    if not transition_errors:
        return 0.0

    return sum(transition_errors) / len(transition_errors)

if __name__ == "__main__":

    edtt_f1 = transition_edtt_f1("LSTM_BASE_0.8831.pth")

    print("Transition EDTT F1:")
    print(edtt_f1)