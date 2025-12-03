import torch
from data_download import BlockDataset
import sys
import os
from main import train_and_predict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.plot import interactive_plot

block_size =12

if __name__ == "__main__":
    train_data = BlockDataset("../../data/pretrained/tens_sequence/tens_concatenated.txt",block_size)
    test_data = BlockDataset("../../data/pretrained/tens_sequence/tens_test.txt",block_size)
    
    y_pred,y_true,X = train_and_predict()
    