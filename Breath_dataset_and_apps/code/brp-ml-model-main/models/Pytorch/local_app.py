import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append("/workspaces/Breath-Project/Breath_dataset_and_apps/code/brp-ml-model-main/models/Pytorch")
from Visualisation.show_mistakes import mistakes_table
from Visualisation.confusion_matrix import conf_Matrix
from main import train_and_predict
from Visualisation.show import plot_results

st.title("ML Model Tester")

if "page" not in st.session_state:
    st.session_state.page = None

def starting_page():

    with st.container():
        st.write("Choose function")
        col11,col12 = st.columns(2)
        col21,col22 =st.columns(2)

        if(col12.button("show")):
             st.session_state.page = "show"
      
            
        if col11.button("main"):
            st.session_state.page = "main"

        if col21.button("show_mistakes"):
            st.session_state.page = "mistakes"

        if col22.button("confusion_matrix"):
            st.session_state.page = "matrix"

        if st.session_state.page == "mistakes":
            mistakes()

        elif st.session_state.page == "matrix":
            matrix()

        elif st.session_state.page == "main":
            conf_hiperparametres()

        elif st.session_state.page == "show":
            interactive_plot()


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
    
def conf_hiperparametres():
    block_size = st.number_input(
        "Block size",
        min_value=1,
        step=1
    )

    batch_options = [2**i for i in range(0, 9)]  # 1,2,4,...,256
    batch_size = st.selectbox(
        "Batch size",
        batch_options
    )

    # target – liczby od 0 do 5
    target = st.selectbox(
        "Target class",
        list(range(0, 6))
    )
    hidden_units = st.number_input(
        "Hidden units",
        min_value=2,
        max_value=64,
        step=2,
        value=64
    )
    output_shape = st.number_input(
        "Hidden units",
        min_value=4,
        max_value=4,
        step=2,
        value=4
    )

    models = [
        "LSTM_ATTENTION",
        "LSTM_STACKED",
        "GruModel",
        "LSTM_BIDIRECTIONAL",
        "GRUAttentionModel",
        "LSTM_BASE",
        "LSTM_DROPOUT",
        "LSTM_CONV1",
        "Transformer"
    ]

        
    model_type = st.selectbox("Choose model:", models)

    num_epchos = st.number_input(
        "num_epchos",
        min_value=1,
        max_value=512,
        step=1,
        value=32
    )
    learning_rate = st.number_input(
    "Learning rate",
    min_value=0.0001,
    max_value=1.0,
    value=0.01,
    step=0.0001,
    format="%.4f"
    )
    pom =[
        "nie",
        "tak"
    ]
    start = st.selectbox("Start", pom)

    if(start!='nie'):
        if st.button("Train Configuration"):
            train_and_predict(block_size,batch_size,target,hidden_units,output_shape,model_type,learning_rate,num_epchos)
            st.session_state.main_clicked=False


def interactive_plot():
    path = Path("saved_models/")
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"]+files
    selected_file = st.selectbox("Choose saved model:", files)
    
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    if selected_file != "--- choose model ---":
        if st.button("Run Interactive Analysis"):
            st.session_state.analysis_started = True
            st.session_state.selected_model = selected_file
        if st.session_state.analysis_started:
                plot_results(f"saved_models/{st.session_state.selected_model}")

                

starting_page()