import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append("/workspaces/Breath-Project/Breath_dataset_and_apps/code/brp-ml-model-main/models/Pytorch")
from Visualisation.show_mistakes import mistakes_table
from Visualisation.confusion_matrix import conf_Matrix
from main import train_and_predict
from Visualisation.show import plot_results
from Visualisation.error_statistic import count_error
from Visualisation.error_statistic import avg_sizeof_error
from Visualisation.error_statistic import avg_position_error
from Visualisation.error_statistic import transitions
from Visualisation.error_statistic import avg_min_max_class_lenght


st.title("ML Model Tester")

if "page" not in st.session_state:
    st.session_state.page = None

def starting_page():

    with st.container():
        st.write("Choose function")
        col11,col12 = st.columns(2)
        col21,col22 =st.columns(2)
        col31,col32 = st.columns(2)

        if(col12.button("show")):
             st.session_state.page = "show"
      
            
        if col11.button("main"):
            st.session_state.page = "main"

        if col21.button("show_mistakes"):
            st.session_state.page = "mistakes"

        if col22.button("confusion_matrix"):
            st.session_state.page = "matrix"
            
        if col31.button("error_statistic"):
            st.session_state.page = "error_statistic"

        if st.session_state.page == "mistakes":
            mistakes()

        elif st.session_state.page == "matrix":
            matrix()

        elif st.session_state.page == "main":
            conf_hiperparametres()

        elif st.session_state.page == "show":
            interactive_plot()
            
        elif st.session_state.page == "error_statistic":
            statistic_plot()


def mistakes():
    path = Path("models/saved_models/")
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"]+files
    selected_file = st.multiselect("Choose saved model:", files)
    
    if "--- choose model ---" not in selected_file:
        if st.button("Run Mistakes Analysis"):
            for model_name in selected_file:

                st.subheader(f"Model: {model_name}")

                labels, values =mistakes_table(f"models/saved_models/{model_name}")
                fig, ax = plt.subplots()
                ax.bar(labels, values)
                ax.set_title(f"Mistakes for {model_name}")
                st.pyplot(fig)
            st.session_state.show_mistakes_clicked = False
def matrix():
    path = Path("models/saved_models/")
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"]+files
    selected_file = st.multiselect("Choose saved model:", files)
    
    if "--- choose model ---" not in selected_file:
        if st.button("Confusion Matrix Analysis"):
            for model_name in selected_file:

                st.subheader(f"Model: {model_name}")
                disp =conf_Matrix(f"models/saved_models/{model_name}")
                fig, ax = plt.subplots(figsize=(6,6))
                disp.plot(ax=ax, cmap='Blues', colorbar=True)


                plt.title(f"Confusion Matrix for {model_name}")
                plt.xlabel("Predicted")
                plt.ylabel("Real")
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
        max_value=512,
        step=1,
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
    min_value=0.00000000001,
    max_value=1.0,
    value=0.01,
    step=0.00000000001,
    format="%.12f"
    )
    loss = [
        "CrossEntropyLoss",
        "FocalLossAdaptive",
        "FocalLoss" 
    ]

        
    loss_type = st.selectbox("Choose loss:", loss)
    optimizer = [
        "Adam"
    ]

        
    optimizer_type = st.selectbox("Choose optimizer:", optimizer)
    
    dropout = st.number_input(
    "dropout",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
    format="%.2f"
    )
    num_layers = st.number_input(
        "num_layers",
        min_value=1,
        max_value=5,
        step=1,
        value=1
    )
    powers_of_two = [2**i for i in range(0, 10)]  # 1,2,4,...,256
    n_head = st.selectbox(
        "n_head",
        powers_of_two,
        index=powers_of_two.index(2)
    )
    dim_feedforward= st.selectbox(
        "dim_feedforward",
        powers_of_two,
        index=powers_of_two.index(64)
    )
    d_model= st.selectbox(
        "d_model",
        powers_of_two,
        index=powers_of_two.index(32)
    )
    
    
    
    pom =[
        "nie",
        "tak"
    ]
    start = st.selectbox("Start", pom)

    if(start!='nie'):
        if st.button("Train Configuration"):
            train_and_predict(block_size,batch_size,target,hidden_units,output_shape,model_type,learning_rate,num_epchos,loos_type=loss_type,optimizer_type=optimizer_type,dropout=dropout,num_layers=num_layers,nhead=n_head,dim_feedforward=dim_feedforward,d_model=d_model)
            st.session_state.main_clicked=False


def interactive_plot():
    path = Path("models/saved_models/")
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"]+files
    selected_file = st.multiselect("Choose saved model:", files)
    
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    if "--- choose model ---" not in selected_file:
        if st.button("Run Interactive Analysis"):
                st.session_state.analysis_started = True
                st.session_state.selected_model = selected_file
        if st.session_state.analysis_started:
            for model in  st.session_state.selected_model:
                plot_results(f"models/saved_models/{model}")

def statistic_plot():
    path = Path("models/saved_models/")
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"]+files
    selected_file = st.multiselect("Choose saved model:", files)
    
    
    if "--- choose model ---" not in selected_file:
        if st.button("Confusion Matrix Analysis"):
            for model_name in selected_file:
                count =count_error(f"models/saved_models/{model_name}")
                avg_sizeof = avg_sizeof_error(f"models/saved_models/{model_name}")
                avg_pos = avg_position_error(f"models/saved_models/{model_name}",1,2)
                mistakes = transitions(f"models/saved_models/{model_name}")
                lenght_info = avg_min_max_class_lenght(f"models/saved_models/{model_name}")
                
                st.subheader(f"Model {model_name}")

                st.write(f"Średnia pozycja błędu: {avg_pos}")
                st.write(f"Liczba błędów: {count}")
                st.write(f"sredni rozmiar bedu: {avg_sizeof}")
                st.write(f"transitions {mistakes}")
                st.write(f"Rozmiary class {lenght_info}")
            st.session_state.error_statistic = False   
                               

starting_page()