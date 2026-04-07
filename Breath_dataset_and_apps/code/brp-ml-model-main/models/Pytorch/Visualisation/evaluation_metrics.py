import torch
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error
from scripts.error_tolerance import RR_error

def standard_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    correct_prediction=0
    
    for i in range(n_samples):
        if y_pred[i]==y_true[i]:
            correct_prediction+=1
        
    return (correct_prediction/n_samples) *100

def with_epsilon_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    correct_prediction=0
    
    for i in range(n_samples):
        if y_pred[i]==y_true[i]:
            correct_prediction+=1
        elif acceptable_error(y_pred,y_true,i,2): #EPSILON = 2  data on boundaries with sizeof <= 2 is not included in error
            correct_prediction+=1
        
    return (correct_prediction/n_samples) *100

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
         
    if predicted_transitions == 0:
        return 0.0

    return (true_transitions/predicted_transitions) *100

def cycle_accuracy(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)
    
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
                    if RR_error(cycle_pred,cycle_true,1):
                         pred_cycle+=1
                if y_true[j]==2 and start_inhale==False:
                    true_cycle+=1
                    i=j+1
                    break
                if j==n_samples-1:
                    i=n_samples+1     
        else:
            i+=1
    return (pred_cycle/true_cycle) *100


