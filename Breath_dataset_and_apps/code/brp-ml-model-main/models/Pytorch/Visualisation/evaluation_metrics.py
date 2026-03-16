import torch
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error


def standard_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    correct_prediction=0
    
    for i in range(n_samples):
        if y_pred[i]==y_true[i]:
            correct_prediction+=1
        
    return (n_samples/correct_prediction) *100

def with_epsilon_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    correct_prediction=0
    
    for i in range(n_samples):
        if y_pred[i]==y_true[i]:
            correct_prediction+=1
        elif acceptable_error(y_pred,y_true,i,2): #EPSILON = 2  data on boundaries with sizeof <= 2 is not included in error
            correct_prediction+=1
        
    return (n_samples/correct_prediction) *100

def number_of_transitions_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    
    true_transitions =0
    predicted_transitions=0
    for i in range(1,n_samples):
        if y_true[i-1]!=y_true[i]:
            true_transitions+=1
        if y_pred[i-1]!=y_pred[i]:
            predicted_transitions+=1
         
    return true_transitions/predicted_transitions

def cycle_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    
    i=0
    pred_cycle=0
    true_cycle=0
    while i < n_samples:
        if y_true[i] == 2:
            is_cycle = True
            start_inhale = True
            for j in range(i,n_samples):
                if y_true[j] != y_pred[j]:
                    if not acceptable_error(y_pred,y_true,j,2):
                        is_cycle = False
                if y_true[j]!=2:
                    start_inhale = False
                if y_true[j]==2 and start_inhale==False and is_cycle==True:
                    pred_cycle+=1
                elif y_true[j]==2 and start_inhale==False:
                    true_cycle+=1
                    i=j
                    break     
        else:
            i+=1
    return (pred_cycle/true_cycle) *100

if __name__ == "__main__":
    print(cycle_accuracy("models/saved_models/JuliaLabel"))

