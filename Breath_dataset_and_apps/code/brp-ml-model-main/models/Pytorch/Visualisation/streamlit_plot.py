import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def streamlit_plot_function(features, labels_predicted, labels_actual,key_prefix, window_size=150, title="Interactive plot"):
   
       
    if len(labels_actual.shape) > 1 and labels_actual.shape[1] == 3:
        labels_actual = labels_actual.argmax(axis=1)

    max_index = len(features) - window_size

    start_index = st.slider(
        "Start index",
        min_value=0,
        max_value=max_index,
        step=10,
        value=0,
        key=f"{key_prefix}_slider"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,6))

    def plot(ax, labels, predicted=True):
        
    
     
        colors = [
            "red" if m == 0 else
            "green" if m == 1 else
            "blue" if m == 2 else
            "yellow" if m == 3 else
            "pink"
            for m in labels
        ]

        ax.scatter(
            range(start_index, start_index + window_size),
            features[start_index:start_index + window_size],
            color=colors[start_index:start_index + window_size],
        )

        ax.plot(
            range(start_index, start_index + window_size),
            features[start_index:start_index + window_size],
            color="black",
            alpha=0.4,
        )

        ax.set_xlim(start_index, start_index + window_size)
        ax.set_ylabel("Predicted" if predicted else "Actual")

    print(labels_actual)
    plot(ax1, labels_predicted, predicted=True)
    plot(ax2, labels_actual, predicted=False)

    plt.suptitle(title)
    st.pyplot(fig)