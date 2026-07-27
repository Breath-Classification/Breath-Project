import torch
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error

def avg_sizeof_error(model_path):
    
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    
    actual_size =0
    error_lengh =0
    error_count =0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            actual_size+=1
        else:
            if actual_size > 0:
                error_count+=1
                error_lengh+=actual_size
                actual_size=0
    if actual_size>0:
        error_count+=1
        error_lengh+=actual_size
    if error_count == 0:
        return 0
    
    return error_lengh/error_count
            
def avg_position_error(model_path, true_class, wrong_class):
    y_pred, y_true, _, _ = load_model_and_predict(model_path)

    positions = []
    i = 0
    N = len(y_true)

    while i < N:
      
        if y_true[i] == true_class:
            start = i
            
            while i < N and y_true[i] == true_class:
                i += 1
                
            end = i 
            segment_length = end - start
            for j in range(start, end):
                if y_pred[j] == wrong_class and segment_length>1:
                    pos_percent = ((j - start) / (segment_length - 1)) * 100
                    positions.append(pos_percent)
        else:
            i += 1

    if len(positions) == 0:
        return None

    return sum(positions) / len(positions)
        

def count_error(model_path):
    
    y_pred,y_true,X,_ = load_model_and_predict(model_path)
    
    actual_size =0

    error_count =0
    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            actual_size+=1
        else:
            if actual_size > 0:
                error_count+=1
                actual_size=0
    if actual_size>0:
        error_count+=1
  
    return error_count
def transitions(model_path):
    y_pred,y_true,X,_ = load_model_and_predict(model_path)
    
    mistakes = {
    "blue-green": 0,
    "green-blue":0,
    "blue-red": 0,
    "red-blue":0,
    "blue-yellow": 0,
    "yellow-blue":0,
    "green-red": 0,
    "red-green":0,
    "green-yellow": 0,
    "yellow-green":0,
    "yellow-red": 0,
    "red-yellow":0
    
    }
    for i in range(len(y_pred)-1): 
        previous = y_pred[i]
        next = y_pred[i+1]
        if previous == 0 and next == 1:
            mistakes["green-red"] += 1
        elif previous == 1 and next == 0:
            mistakes["red-green"] += 1
        elif previous == 0 and next == 2:
            mistakes["blue-red"] += 1
        elif previous == 2 and next == 0:
            mistakes["red-blue"] += 1
        elif previous == 0 and next == 3:
            mistakes["yellow-red"] += 1
        elif previous == 3 and next == 0:
            mistakes["red-yellow"] += 1
        elif previous == 1 and next == 2:
            mistakes["blue-green"] += 1
        elif previous == 2 and next == 1:
            mistakes["green-blue"] += 1
        elif previous == 1 and next == 3:
            mistakes["green-yellow"] += 1
        elif previous == 3 and next == 1:
            mistakes["yellow-green"] += 1
        elif previous == 2 and next == 3:
            mistakes["blue-yellow"] += 1
        elif previous == 3 and next == 2:
            mistakes["yellow-blue"] += 1
    return mistakes

def count_avg_min_max(true_class,ClassMin, ClassMax, ClassSum, ClassCount,counter):
    ClassSum[true_class]+=counter
    ClassCount[true_class]+=1
    
    if ClassMin[true_class]>counter:
        ClassMin[true_class]=counter
        
    if ClassMax[true_class]<counter:
        ClassMax[true_class]=counter
    
def avg_min_max_class_lenght(model_path):
   # y_pred,y_true,X,_ = load_model_and_predict(model_path)
    y_pred,y_true = erase_illegal_transition(model_path)
    ClassMin = {
        "red": 100, 
        "blue": 100,
        "green": 100,
        "yellow":100
    }
    ClassMax = {
        "red": -1, 
        "blue": -1,
        "green": -1,
        "yellow":-1
    }
    ClassAvg = {
        "red": 0, 
        "blue": 0,
        "green": 0,
        "yellow":0
    }
    ClassSum= {
        "red": 0, 
        "blue": 0,
        "green": 0,
        "yellow":0
    }
    ClassCount={
        "red": 0, 
        "blue": 0,
        "green": 0,
        "yellow":0
    }
    
    
    counter =1
    for i in range(1,len(y_pred)):
        if y_true[i-1]==y_true[i]:
            counter +=1 #count lenght of class
        else:
            if y_true[i-1] == 0: #breath out
                count_avg_min_max("red",ClassMin,ClassMax,ClassSum,ClassCount,counter)
                    
            elif y_true[i-1] == 1: #out no breath
                count_avg_min_max("green",ClassMin,ClassMax,ClassSum,ClassCount,counter)
                    
            elif y_true[i-1] == 2: #breath in 
                count_avg_min_max("blue",ClassMin,ClassMax,ClassSum,ClassCount,counter)
                    
            elif y_true[i-1] == 3: #in no breath
                count_avg_min_max("yellow",ClassMin,ClassMax,ClassSum,ClassCount,counter)
            counter =1 #clean counter new sequence
     
    ClassAvg["red"] = ClassSum["red"] / ClassCount["red"]
    ClassAvg["blue"] = ClassSum["blue"] / ClassCount["blue"]
    ClassAvg["green"] = ClassSum["green"] / ClassCount["green"]
    ClassAvg["yellow"] = ClassSum["yellow"] / ClassCount["yellow"]
            
    return ClassMin,ClassMax,ClassAvg


def numer_of_segments(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)

    tab = [0 for i in range(6)]
    counter = 1
    for i in range(1,len(y_pred)):
        if y_true[i-1]==y_true[i]:
            counter +=1
        else:
            if counter < 6:
                tab[counter] += 1
            counter = 1
            tab[0]+=1
    return tab 

def erase_illegal_transition(model_path):
    y_pred,y_true,_,_ = load_model_and_predict(model_path)
    n_samples = len(y_pred)

    
    for i in range(1,len(y_pred)):
        if (y_pred[i-1]==3 and y_pred[i] == 1) or (y_pred[i-1]==1 and y_pred[i] == 3):
            y_pred[i]=y_pred[i-1]
    return y_pred,y_true