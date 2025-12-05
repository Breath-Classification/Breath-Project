import torch
import matplotlib.pyplot as plt 
from main import train_and_predict


if __name__ == "__main__":
    y_pred,y_true,X = train_and_predict()
    mistakes = {
    "blue-green": 0,
    "blue-red": 0,
    "blue-yellow": 0,
    "green-red": 0,
    "green-yellow": 0,
    "yellow-red": 0
    }
    for i in range(len(y_pred)):
        true = y_true[i]
        pred = y_pred[i]
    
        if true != pred:
            if true == 0 and pred == 1 or true == 1 and pred == 0:
                mistakes["green-red"] += 1
            elif true == 0 and pred == 2 or true == 2 and pred == 0:
                mistakes["blue-red"] += 1
            elif true == 0 and pred == 3 or true == 3 and pred == 0:
                mistakes["yellow-red"] += 1
            elif true == 1 and pred == 2 or true == 2 and pred == 1:
                mistakes["blue-green"] += 1
            elif true == 1 and pred == 3 or true == 3 and pred == 1:
                mistakes["green-yellow"] += 1
            elif true == 2 and pred == 3 or true == 3 and pred == 2:
                mistakes["blue-yellow"] += 1
    labels = list(mistakes.keys())
    values = list(mistakes.values())
    plt.bar(labels,values)
    plt.show()