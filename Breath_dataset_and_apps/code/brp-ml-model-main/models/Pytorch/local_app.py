import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append("/workspaces/Breath-Project/Breath_dataset_and_apps/code/brp-ml-model-main/models/Pytorch")
from show_mistakes import mistakes_table
from confusion_matrix import conf_Matrix

st.title("ML Model Tester")

if "show_mistakes_clicked" not in st.session_state:
    st.session_state.show_mistakes_clicked = False
if "confusion_matrix_clicked" not in st.session_state:
    st.session_state.confusion_matrix_clicked = False

def starting_page():

    with st.container():
        st.write("Choose function")
        col11,col12 = st.columns(2)
        col21,col22 =st.columns(2)

        if(col11.button("main")):
            st.write("main")
        if(col12.button("show")):
            st.write("show")
        if(col21.button("show_mistakes")):
            st.session_state.show_mistakes_clicked = True
        if(col22.button("confusion_matrix")):
            st.session_state.confusion_matrix_clicked = True
            
        if st.session_state.show_mistakes_clicked:
            mistakes()
        if st.session_state.confusion_matrix_clicked:
            matrix()

def mistakes():
    path = Path("saved_models/")
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"]+files
    selected_file = st.selectbox("Choose saved model:", files)
    
    if selected_file != "--- choose model ---":
        if st.button("Run Mistakes Analysis"):
            labels, values =mistakes_table(f"saved_models/{selected_file}")
            fig, ax = plt.subplots()
            ax.bar(labels, values)
            st.pyplot(fig)
            st.session_state.show_mistakes_clicked = False
def matrix():
    path = Path("saved_models/")
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"]+files
    selected_file = st.selectbox("Choose saved model:", files)
    
    if selected_file != "--- choose model ---":
        if st.button("Confusion Matrix Analysis"):
            disp =conf_Matrix(f"saved_models/{selected_file}")
            fig, ax = plt.subplots(figsize=(6,6))
            disp.plot(ax=ax, cmap='Blues', colorbar=True)


            plt.title("LSTM CONV1")
            plt.xlabel("Przewidziana klasa")
            plt.ylabel("Rzeczywista klasa")
            st.pyplot(fig)
            st.session_state.confusion_matrix_clicked = False
    


starting_page()