from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from main import load_model_and_predict
import numpy as np
from scripts.error_tolerance import acceptable_error


def conf_Matrix(model_path):

    y_pred, y_true, _,_ = load_model_and_predict(model_path)


    label_mapping = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4:'pink'}


    # This add aceptable error
    for i in range(len(y_true)):
            if(y_true[i]!=y_pred[i] and acceptable_error(y_pred,y_true,i,2)== True):
                y_pred[i]=4 


    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[label_mapping[i] for i in labels_present]
    )

    return disp
    '''
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)


    plt.title("LSTM CONV1")
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Rzeczywista klasa")

    plt.savefig("Matrix/LSTM_CONV1_l2_30_test.png")
    plt.show()
    '''
