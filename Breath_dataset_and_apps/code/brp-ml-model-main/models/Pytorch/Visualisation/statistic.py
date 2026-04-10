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

if __name__ == "__main__":
     acc = epsilon_accuracy("")