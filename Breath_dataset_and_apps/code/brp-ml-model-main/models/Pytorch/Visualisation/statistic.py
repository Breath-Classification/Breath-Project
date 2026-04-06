import torch
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error

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

def precision(model_path, actual_class):
    return

def recall(model_path, actual_class):
    return

def f_scale(precision, recall):
    return

if __name__ == "__main__":
     acc = epsilon_accuracy("")