import torch
from data_download import BlockDataset
import sys
import os
from main import train_and_predict
from main import load_model_and_predict
from scripts.error_tolerance import acceptable_error
from Visualisation.streamlit_plot import streamlit_plot_function

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.plot import interactive_plot 


#You can use two plot scripts streamlit_plot is compatibile with Streamlit and web aplication
# interactive plot works with matplotib


def plot_results(model_path):

    y_pred,y_true,X,_ = load_model_and_predict(model_path)
    
    
    
    #X = X[:, :6] # zrobione dla pochodnych #usun komentarze jezeli sliding window

    #X= X[:, -1] # size
    
    
   
    streamlit_plot_function(X,y_pred,y_true,key_prefix=f"{model_path}original")
    
    for i in range(len(y_true)):
        if(y_true[i]!=y_pred[i] and acceptable_error(y_pred,y_true,i,2)== True):
            y_pred[i]=4 # acceptable error class
    streamlit_plot_function(X,y_pred,y_true, key_prefix=f"{model_path} with error tolerance")