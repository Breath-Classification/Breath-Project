from pathlib import Path
import matplotlib.pyplot as plt
import re
import sys
import argparse
import types


import streamlit as st


sys.path.append("/workspaces/Breath-Project/Breath_dataset_and_apps/code/brp-ml-model-main/models/Pytorch")
from Visualisation.show_mistakes import mistakes_table
from Visualisation.confusion_matrix import conf_Matrix
from main import train_and_predict
from Visualisation.show import plot_results
from Visualisation.error_statistic import count_error
from Visualisation.error_statistic import transitions
from Visualisation.error_statistic import numer_of_segments
from Visualisation.error_statistic import avg_sizeof_error
from Visualisation.error_statistic import avg_position_error
from Visualisation.evaluation_metrics import standard_accuracy
from Visualisation.evaluation_metrics import with_epsilon_accuracy
from Visualisation.evaluation_metrics import cycle_accuracy
from Visualisation.evaluation_metrics import number_of_transitions_accuracy
from Visualisation.statistic import epsilon_accuracy
from Visualisation.statistic import epsilon_RR_accuracy
from Visualisation.statistic import precision
from Visualisation.statistic import recall
from Visualisation.statistic import f_scale
from Visualisation.statistic import transition_edtt_f1

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
SAVED_MODELS_DIR = PROJECT_DIR / "data" / "Models" / "saved_models"
MODEL_LABEL_OVERRIDES = {
    "LSTM_MIX_0.9514": "LSTM_PHYSIOLOGICAL",
}
LSTM_COMPARISON_MODELS = [
    ("LSTM_BASE", "BlockDataset/many_to_one/LSTM_BASE_0.9399.pth"),
    ("LSTM_DROPOUT", "BlockDataset/many_to_one/LSTM_DROPOUT_0.9375.pth"),
    ("LSTM_STACKED", "SequenceBlockDataset/Window/NoCRF/LSTM_STACKED_0.9465.pth"),
    ("LSTM_BIDIRECTIONAL", "BlockDataset/many_to_one/LSTM_BIDIRECTIONAL_0.9333.pth"),
    ("LSTM_CONV1", "BlockDataset/many_to_one/LSTM_CONV1_0.9441.pth"),
    ("LSTM_ATTENTION", "BlockDataset/many_to_one/LSTM_ATTENTION_0.9431.pth"),
    ("LSTM_PHYSIOLOGICAL", "LSTM_MIX_0.9514.pth"),
]

st.title("ML Model Tester")

if "page" not in st.session_state:
    st.session_state.page = None


def parse_model_filename(model_path):
    match = re.match(r"(.+?)_(?:S)?(\d+\.\d+)$", model_path.stem)
    if not match:
        return model_path.stem, 0.0
    return match.group(1), float(match.group(2))


def model_display_name(model_path):
    return MODEL_LABEL_OVERRIDES.get(model_path.stem, parse_model_filename(model_path)[0])


def model_path_from_name(model_name):
    return str(SAVED_MODELS_DIR / model_name)


def find_best_models_by_group():
    best_models = {}

    for model_path in SAVED_MODELS_DIR.rglob("*.pth"):
        if model_path.parent == SAVED_MODELS_DIR:
            continue

        model_name, score = parse_model_filename(model_path)
        if model_name is None:
            continue

        group_name = str(model_path.parent.relative_to(SAVED_MODELS_DIR))
        current_best = best_models.setdefault(group_name, {})
        saved_model = current_best.get(model_name)

        if saved_model is None or score > saved_model["score"]:
            current_best[model_name] = {
                "path": model_path,
                "score": score,
            }

    return best_models


def calculate_accuracy_metrics(model_path):
    model_path_str = str(model_path)
    return {
        "standard accuracy": standard_accuracy(model_path_str),
        "epsilon accuracy": with_epsilon_accuracy(model_path_str),
        "transition accuracy": number_of_transitions_accuracy(model_path_str),
        "transition EDTT F1": transition_edtt_f1(model_path_str),
        "cycle accuracy": cycle_accuracy(model_path_str),
    }


def calculate_error_metrics(model_path):
    model_path_str = str(model_path)
    return {
        "avg_sizeof_error": float(avg_sizeof_error(model_path_str)),
        "avg_position_error": float(avg_position_error(model_path_str, 1, 2) or 0.0),
        "count_error": float(count_error(model_path_str)),
        "mistakes" : transitions(model_path_str),
        "number_of_segments" : numer_of_segments(model_path_str),
    }


def format_metric_value(value):
    return f"{value:.2f}"


def format_ratio_metric_value(value):
    return f"{value * 100:.2f}"



def plot_best_models_metrics(group_name, metrics_by_model):
    metric_names = list(next(iter(metrics_by_model.values())).keys())
    model_names = list(metrics_by_model.keys())
    positions = list(range(len(model_names)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.4), 6))

    for index, metric_name in enumerate(metric_names):
        values = [metrics_by_model[model_name][metric_name] for model_name in model_names]
        offset = (index - (len(metric_names) - 1) / 2) * width
        ax.bar([position + offset for position in positions], values, width=width, label=metric_name)

    ax.set_title(f"Best models statistics for {group_name}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_epsilon_accuracy_curves(group_name, epsilon_by_model, title="Epsilon accuracy by tolerance"):
    fig, ax = plt.subplots(figsize=(10, 6))

    max_epsilon = 0
    for model_name, accuracies in epsilon_by_model.items():
        epsilons = list(range(len(accuracies)))
        max_epsilon = max(max_epsilon, len(accuracies))
        ax.plot(epsilons, accuracies, marker="o", linewidth=2, label=model_name)

    ax.set_title(f"{title} for {group_name}")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy [%]")
    ax.set_xticks(list(range(max_epsilon or 1)))
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    return fig

def plot_epsilon_accuracy(model_name, accuracies, title="Epsilon Accuracy"):
    """
    accuracies - lista wartości accuracy dla kolejnych epsilonów
                 np. [84.1, 87.3, 90.5, 92.2, ...]
    """

    epsilons = list(range(len(accuracies)))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epsilons,
        accuracies,
        marker="o",
        linewidth=2,
        label=model_name
    )

    ax.set_title(f"{title} - {model_name}")
    ax.set_xlabel("Epsilon Size")
    ax.set_ylabel("Epsilon Accuracy (%)")

    ax.set_xticks(epsilons)
    ax.set_ylim(0, 100)

    ax.grid(True, linestyle="--", alpha=0.4)

    # Możesz usunąć legendę, skoro jest tylko jedna linia
    # ax.legend()

    fig.tight_layout()

    return fig

def plot_single_error_metric(group_name, metric_name, metrics_by_model):
    model_names = list(metrics_by_model.keys())
    values = [metrics_by_model[model_name][metric_name] for model_name in model_names]
    positions = list(range(len(model_names)))

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.2), 5))
    ax.bar(positions, values)
    ax.set_title(f"{metric_name} - {group_name}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def starting_page():

    with st.container():
        st.write("Choose function")
        col11, col12 = st.columns(2)
        col21, col22 = st.columns(2)
        col31, col32 = st.columns(2)
        col41,col42 =st.columns(2)
        if col12.button("show"):
            st.session_state.page = "show"

        if col11.button("main"):
            st.session_state.page = "main"

        if col21.button("show_mistakes"):
            st.session_state.page = "mistakes"

        if col22.button("confusion_matrix"):
            st.session_state.page = "matrix"

        if col31.button("error_statistic"):
            st.session_state.page = "error_statistic"

        if col32.button("accuracy_statistic"):
            st.session_state.page = "accuracy_statistic"
        
        if col41.button("single_model_accuracy"):
            st.session_state.page = "single_model_accuracy"

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

        elif st.session_state.page == "accuracy_statistic":
            accuracy()
        elif st.session_state.page == "single_model_accuracy":
            single_model_accuracy()

def single_model_accuracy():
    st.subheader("Single Model Accuracy")

    model_files = sorted(
        SAVED_MODELS_DIR.rglob("*.pth")
    )

    if not model_files:
        st.warning(f"No .pth models found in {SAVED_MODELS_DIR}.")
        return

    model_options = {
        str(
            model_path.relative_to(SAVED_MODELS_DIR)
        ): model_path
        for model_path in model_files
    }

    selected_model = st.selectbox(
        "Choose model:",
        list(model_options.keys())
    )

    model_path = model_options[selected_model]

    st.write(f"Selected model:")
    st.code(str(model_path))

    if st.button("Calculate metrics"):
        with st.spinner("Calculating metrics..."):

            metrics = calculate_accuracy_metrics(model_path)

        st.subheader("Model metrics")

        summary_row = {
            "model": model_display_name(model_path),
            "path": selected_model,
            "best saved score": format_metric_value(
                parse_model_filename(model_path)[1] * 100
            ),
            **{
                key: format_metric_value(value)
                for key, value in metrics.items()
            },
        }

        st.dataframe(
            [summary_row],
            use_container_width=True
        )

def mistakes():
    path = SAVED_MODELS_DIR
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"] + files
    selected_file = st.multiselect("Choose saved model:", files)

    if "--- choose model ---" not in selected_file:
        if st.button("Run Mistakes Analysis"):
            for model_name in selected_file:

                st.subheader(f"Model: {model_name}")

                labels, values = mistakes_table(model_path_from_name(model_name))
                fig, ax = plt.subplots()
                ax.bar(labels, values)
                ax.set_title(f"Mistakes for {model_name}")
                st.pyplot(fig)
            st.session_state.show_mistakes_clicked = False


def matrix():
    path = SAVED_MODELS_DIR
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"] + files
    selected_file = st.multiselect("Choose saved model:", files)

    if "--- choose model ---" not in selected_file:
        if st.button("Confusion Matrix Analysis"):
            for model_name in selected_file:

                st.subheader(f"Model: {model_name}")
                disp = conf_Matrix(model_path_from_name(model_name))
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(ax=ax, cmap="Blues", colorbar=True)

                plt.title(f"Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Real")
                st.pyplot(fig)
            st.session_state.confusion_matrix_clicked = False


def conf_hiperparametres():
    block_size = st.number_input(
        "Block size",
        min_value=1,
        step=1,
    )

    batch_options = [2 ** i for i in range(0, 9)]
    batch_size = st.selectbox(
        "Batch size",
        batch_options,
    )

    target = st.selectbox(
        "Target class",
        list(range(0, 6)),
    )
    hidden_units = st.number_input(
        "Hidden units",
        min_value=2,
        max_value=512,
        step=1,
        value=64,
    )
    output_shape = st.number_input(
        "Hidden units",
        min_value=4,
        max_value=4,
        step=2,
        value=4,
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
        "Transformer",
    ]

    model_type = st.selectbox("Choose model:", models)

    num_epchos = st.number_input(
        "num_epchos",
        min_value=1,
        max_value=512,
        step=1,
        value=32,
    )
    learning_rate = st.number_input(
        "Learning rate",
        min_value=0.00000000001,
        max_value=1.0,
        value=0.01,
        step=0.00000000001,
        format="%.12f",
    )
    loss = [
        "CrossEntropyLoss",
        "FocalLossAdaptive",
        "FocalLoss",
    ]

    loss_type = st.selectbox("Choose loss:", loss)
    optimizer = [
        "Adam",
    ]

    optimizer_type = st.selectbox("Choose optimizer:", optimizer)

    dropout = st.number_input(
        "dropout",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        format="%.2f",
    )
    num_layers = st.number_input(
        "num_layers",
        min_value=1,
        max_value=5,
        step=1,
        value=1,
    )
    powers_of_two = [2 ** i for i in range(0, 10)]
    n_head = st.selectbox(
        "n_head",
        powers_of_two,
        index=powers_of_two.index(2),
    )
    dim_feedforward = st.selectbox(
        "dim_feedforward",
        powers_of_two,
        index=powers_of_two.index(64),
    )
    d_model = st.selectbox(
        "d_model",
        powers_of_two,
        index=powers_of_two.index(32),
    )

    pom = [
        "nie",
        "tak",
    ]
    start = st.selectbox("Start", pom)

    if start != "nie":
        if st.button("Train Configuration"):
            train_and_predict(
                block_size,
                batch_size,
                target,
                hidden_units,
                output_shape,
                model_type,
                learning_rate,
                num_epchos,
                loos_type=loss_type,
                optimizer_type=optimizer_type,
                dropout=dropout,
                num_layers=num_layers,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                d_model=d_model,
            )
            st.session_state.main_clicked = False


def interactive_plot():
    path = SAVED_MODELS_DIR
    files = [f.name for f in path.glob("*.pth")]
    files = ["--- choose model ---"] + files
    selected_file = st.multiselect("Choose saved model:", files)

    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False

    if "--- choose model ---" not in selected_file:
        if st.button("Run Interactive Analysis"):
            st.session_state.analysis_started = True
            st.session_state.selected_model = selected_file
        if st.session_state.analysis_started:
            for model in st.session_state.selected_model:
                plot_results(model_path_from_name(model))


def statistic_plot():
    if st.button("Show best models error statistics"):
        best_models = find_best_models_by_group()

        if not best_models:
            st.warning(f"No grouped models were found in {SAVED_MODELS_DIR}.")
            return

        for group_name, models in sorted(best_models.items()):
            st.subheader(f"Group: {group_name}")

            metrics_by_model = {}
            summary_rows = []

            for model_name, model_info in sorted(models.items()):
                metrics = calculate_error_metrics(model_info["path"])
                metrics_by_model[model_name] = metrics
                summary_rows.append(
                    {
                        "model": model_display_name(model_info["path"]),
                        "mistakes": str(metrics["mistakes"]),
                        
                       # "best saved score": format_metric_value(model_info["score"] * 100),
                        #"avg_sizeof_error": format_metric_value(metrics["avg_sizeof_error"]),
                        #"avg_position_error": format_metric_value(metrics["avg_position_error"]),
                        #"count_error": format_metric_value(metrics["count_error"]),
                        #"number_of_segments": str(metrics["number_of_segments"]),
                        
                    }
                )

            st.dataframe(summary_rows, use_container_width=True)
          

            
            st.write(f"### {model_name}")
            st.write("Mistakes:")
            st.dataframe(metrics["mistakes"])

            '''
            st.write("Number of segments:")
            st.dataframe(metrics["number_of_segments"])
            st.pyplot(plot_single_error_metric(group_name, "avg_sizeof_error", metrics_by_model))
            st.pyplot(plot_single_error_metric(group_name, "avg_position_error", metrics_by_model))
            st.pyplot(plot_single_error_metric(group_name, "count_error", metrics_by_model))
            '''
    
        st.session_state.error_statistic = False


def accuracy():
    if st.button("Show best models statistics"):
        best_models = find_best_models_by_group()

        if not best_models:
            st.warning(f"No grouped models were found in {SAVED_MODELS_DIR}.")
            return

        for group_name, models in sorted(best_models.items()):
            st.subheader(f"Group: {group_name}")

            metrics_by_model = {}
            epsilon_by_model = {}
            epsilon_rr_by_model = {}
            class_metrics_rows = []
            summary_rows = []

            for model_name, model_info in sorted(models.items()):
                metrics = calculate_accuracy_metrics(model_info["path"])
                metrics_by_model[model_name] = metrics
                epsilon_by_model[model_name] = epsilon_accuracy(str(model_info["path"]))
                epsilon_rr_by_model[model_name] = epsilon_RR_accuracy(str(model_info["path"]))
                model_path = str(model_info["path"])
                
                #for cls in range(0, 4):  
                 #   p = precision(model_path, cls)
                  #  r = recall(model_path, cls)
                   # f1 = f_scale(p, r)

                    #class_metrics_rows.append({
                     #   "model": model_name,
                      #  "class": cls,
                       # "precision": format_ratio_metric_value(p),
                        #"recall": format_ratio_metric_value(r),
                        #"f_score": format_ratio_metric_value(f1),
                #    })
                
                summary_rows.append(
                    {
                        "model": model_display_name(model_info["path"]),
                        "best saved score": format_metric_value(model_info["score"] * 100),
                        **{key: format_metric_value(value) for key, value in metrics.items()},
                    }
                )
                

            st.dataframe(summary_rows, use_container_width=True)
            #st.write("Precision / Recall / F-score by class")
            #st.dataframe(class_metrics_rows, use_container_width=True)
            #st.pyplot(plot_best_models_metrics(group_name, metrics_by_model))
            #st.pyplot(plot_epsilon_accuracy_curves(group_name, epsilon_by_model))
            #st.pyplot(plot_epsilon_accuracy_curves(group_name, epsilon_rr_by_model, title="Epsilon RR accuracy by tolerance"))
            #st.pyplot(plot_epsilon_accuracy("LSTM_BASE", epsilon_by_model))

        st.session_state.accuracy_statistic = False


def plot_model_metrics_image(model_path):
    metrics = calculate_accuracy_metrics(model_path)
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar(labels, values, width=0.55)
    ax.set_title("Model comparison")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=25)

    for index, value in enumerate(values):
        ax.text(index, value + 1, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(model_display_name(model_path), y=0.98, fontsize=11)
    fig.tight_layout()
    return fig


def export_model_metrics_image(model_name, output_name=None):
    model_path = SAVED_MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_name = output_name or f"{model_display_name(model_path)}.png"
    output_path = SAVED_MODELS_DIR / output_name
    fig = plot_model_metrics_image(model_path)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def plot_lstm_comparison_image(metrics_by_model):
    metric_names = list(next(iter(metrics_by_model.values())).keys())
    model_names = list(metrics_by_model.keys())
    positions = list(range(len(model_names)))
    width = 0.18
    colors = ["#2563eb", "#16a34a", "#f97316", "#9333ea"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for index, metric_name in enumerate(metric_names):
        values = [metrics_by_model[model_name][metric_name] for model_name in model_names]
        offset = (index - (len(metric_names) - 1) / 2) * width
        bars = ax.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=metric_name,
            color=colors[index % len(colors)],
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("LSTM models comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy [%]")
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(metric_names))
    fig.tight_layout()
    return fig


def export_lstm_comparison_image(output_name="LSTM_MODELS_COMPARISON.png"):
    metrics_by_model = {}

    for display_name, relative_path in LSTM_COMPARISON_MODELS:
        model_path = SAVED_MODELS_DIR / relative_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        metrics_by_model[display_name] = calculate_accuracy_metrics(model_path)

    output_path = SAVED_MODELS_DIR / output_name
    fig = plot_lstm_comparison_image(metrics_by_model)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Local Streamlit app and metrics image exporter.")
    parser.add_argument("--export-model-image", help="Saved model filename, for example LSTM_MIX_0.9514.pth.")
    parser.add_argument("--export-lstm-comparison", action="store_true", help="Export one chart with all selected LSTM models.")
    parser.add_argument("--output-name", help="Output PNG filename. Defaults to the model display name or LSTM_MODELS_COMPARISON.png.")
    return parser.parse_args()


args = parse_args()
if args.export_model_image:
    saved_path = export_model_metrics_image(args.export_model_image, args.output_name)
    print(f"Saved image: {saved_path}")
elif args.export_lstm_comparison:
    saved_path = export_lstm_comparison_image(args.output_name or "LSTM_MODELS_COMPARISON.png")
    print(f"Saved image: {saved_path}")
else:
    starting_page()
