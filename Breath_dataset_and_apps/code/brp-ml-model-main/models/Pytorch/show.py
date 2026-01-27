import torch
from data_download import BlockDataset
import sys
import os
from main import train_and_predict
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.plot import interactive_plot

block_size =30

if __name__ == "__main__":

    y_pred,y_true,X,_ = load_model_and_predict(model_path="saved_models/BaseB.pth")
    
    print(X.size(), "first")
    X = X[:, :6] # zrobione dla pochodnych

    X= X[:, -1] # size
    print(X.size(), "second")
   
    interactive_plot(X,y_pred,y_true)
    
    for i in range(len(y_true)):
        if(y_true[i]!=y_pred[i] and acceptable_error(y_pred,y_true,i,2)== True):
            y_pred[i]=4 # acceptable error class
    interactive_plot(X,y_pred,y_true)