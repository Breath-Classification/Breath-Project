## Introduction

This repository contains a respiratory analysis framework based on artificial intelligence and machine learning. The project is an extension of the work presented in https://www.nature.com/articles/s41597-025-04625-5, which was initially developed using the TensorFlow framework.

The main goal of this project is to provide a more flexible and extensible environment for developing, training, evaluating, and deploying models for respiratory phase recognition. The original TensorFlow implementation was migrated to PyTorch and extended with additional machine learning models, including LSTM-based architectures, Transformers, and a physiology-aware model incorporating knowledge about the respiratory process.

The project also extends the original system with a personalization framework that allows models to be adapted to individual users. Personalization can be performed either locally on a mobile device or remotely using a server-based approach.

The complete system consists of three main components: the machine learning framework, the backend responsible for server-side processing and personalization, and a cross-platform mobile application for Android and iOS.

Note: This repository is currently a work in progress and is under active development. Its structure, implementations, models, and documentation are continuously being extended and improved.

## Main Contributions

The main contributions of this project are:

* **Migration from TensorFlow to PyTorch** – the core functionalities of the original TensorFlow implementation were rewritten and adapted to the PyTorch framework.

* **Extension of the model collection** – additional predictive models were implemented and integrated into the framework, including Transformer-based models, LSTM with Attention, and the proposed physiology-aware model.

* **Physiology-aware framework** – a framework incorporating physiological knowledge into the learning process was developed. Its purpose is to investigate how incorporating information about the respiratory process can improve model performance and provide a better understanding of the analyzed problem.

* **Model personalization** – a framework for adapting the trained model to an individual user was developed. Personalization can be performed through a server-based approach using the backend and mobile application.

* **Mobile application development** – the original application was rewritten and extended to support PyTorch models and the personalization mechanism. The new application provides cross-platform support for Android and iOS, local on-device personalization, and communication with the server for remote personalization.

## System Overview

This repository is one of three repositories that together form the complete respiratory analysis and personalization system.

| Repository           | Description                                                                                                       | Link                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Machine Learning** | Machine learning framework for training, evaluating, and analyzing respiratory phase recognition models.          | [This repository](https://github.com/Breath-Classification/Breath-Project)           |
| **Backend**          | Backend responsible for communication, server-side model personalization, and supporting the mobile application.  | [Backend repository](https://github.com/maciejdrywa/BreathSenseApp-backend)   |
| **Frontend**         | Cross-platform mobile application for Android and iOS, including on-device and server-side model personalization. | [Frontend repository](https://github.com/maciejdrywa/BreathSenseApp-frontend) |

### Machine Learning Repository

This repository contains the main machine learning part of the project. It provides the models, training procedures, evaluation methods, datasets, and tools required for respiratory phase analysis.

For **analysis and training of models using the existing datasets**, only this repository is required. The Backend and Frontend repositories are not necessary for these tasks.

The Backend and Frontend repositories are required when using the **personalization framework**, including model fine-tuning and running the complete mobile application.

This repository contains the main documentation of the machine learning part of the project. More detailed information about the Backend and Frontend components, including their installation, configuration, and usage, can be found in their respective repositories.

A more detailed overview of the complete system architecture will be provided below.

## Repository Structure

The main structure of the repository is organized around the machine learning framework and the application code.

```text
Breath-Project/
├── Breath_dataset_and_apps/
│   └── code/
│       ├── brp-ml-model-main/
│       │   ├── graphs/
│       │   ├── models/
│       │   │   ├── Keras/
│       │   │   └── Pytorch/
│       │   └── scripts/
│       └── brp-app-main/
│
└── ...
```

### Main Directories

| Directory                                                        | Description                                                                                         |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `Breath_dataset_and_apps/code/brp-ml-model-main/`                | Main machine learning project containing the models, experiments, and training scripts.             |
| `Breath_dataset_and_apps/code/brp-ml-model-main/models/`         | Contains the implemented machine learning models, including both Keras and PyTorch implementations. |
| `Breath_dataset_and_apps/code/brp-ml-model-main/models/Keras/`   | Contains the original TensorFlow/Keras implementation.                                              |
| `Breath_dataset_and_apps/code/brp-ml-model-main/models/Pytorch/` | Contains the current PyTorch implementation and the main development of the extended framework.     |
| `Breath_dataset_and_apps/code/brp-app-main/`                     | Contains the mobile application code from the original project.                                     |

The **PyTorch implementation** is the main part of the current machine learning framework. The **Keras implementation** is retained to provide compatibility with and reference to the original work.





## Contributors

| Contributor                | Contributions                                                                                                                                |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Szymon Drywa**           | Development of the physiology-aware framework, migration from TensorFlow to PyTorch, and development of the model personalization framework. |
| **Piotr Lange**            | Design of application architecture, implementation of the server component, communication pipeline, and native Android FT components.development.                               |
| **Maciej Drywa**           | Design of mobile application architecture using React Native and implementation of models on Android and iOS mobile systems.                             |
| **Prof. Julian Szymański** | Scientific, hardware, and analytical support throughout the project.                                                                         |

## Data Availability

The project uses two main sources of respiratory data:

* **Original dataset** – the data used in the original work are available in [https://mostwiedzy.pl/en/open-research-data/respiratory-rythm-phases-classifiction-dataset,40910301641618-0]. These data were used to reproduce and extend the original experiments and to provide compatibility with the previous TensorFlow implementation.

* **New dataset** – additional respiratory data were collected by the authors of this project and are available in [repository/link]. These data were collected using the same general measurement approach and were used to develop and evaluate the extended framework and its models.

Both datasets are used within the project for model training, testing, and evaluation. The original dataset is attributed to the authors of the previous work, while the new dataset was collected as part of the present project.

## Models

The project provides a collection of respiratory phase classification models implemented in TensorFlow and PyTorch. The repository contains both models retained from the original implementation and newly developed architectures.

### Models available in this repository

| Model                          | TensorFlow | PyTorch | Description                                                                                                |
| ------------------------------ | :--------: | :-----: | ---------------------------------------------------------------------------------------------------------- |
| **LSTM**                       |      ✓     |    ✓    | Baseline recurrent neural network model.                                                                   |
| **GRU**                        |      ✓     |    ✓    | Gated recurrent neural network model.                                                                      |
| **LSTM with Attention**        |      —     |    ✓    | LSTM model enhanced with an attention mechanism.                                                           |
| **Bidirectional LSTM**         |      —     |    ✓    | LSTM processing the input sequence in both directions.                                                     |
| **CNN**                        |      —     |    ✓    | Convolutional neural network for respiratory phase classification.                                         |
| **LSTM with Dropout**          |      —     |    ✓    | LSTM model with dropout-based regularization.                                                              |
| **Physiology-Aware Model**     |      —     |    ✓    | Bidirectional LSTM and CNN architecture incorporating physiological constraints into the learning process. |
| **Transformer**                |      —     |    ✓    | Transformer-based sequence classification model.                                                           |
| **Transformer with Attention** |      —     |    ✓    | Transformer architecture incorporating an additional attention mechanism.                                  |
| **Transformer with CRF**       |      —     |    ✓    | Transformer-based model combined with a Conditional Random Field (CRF) layer.                              |

### Previous Models

Other models from the original project are available in a separate repository. These include traditional machine learning approaches such as:

* **Random Forest**
* **Support Vector Machine (SVM)**

The original LSTM and GRU implementations are also retained in this repository in their TensorFlow versions, alongside their PyTorch implementations.

## Technologies

The project uses the following technologies and tools:

* **Python** – primary programming language for the machine learning framework.
* **PyTorch** – main framework used for the development and training of new models.
* **TensorFlow** – used for the original implementations of LSTM and GRU models.
* **Docker** – used to provide a reproducible development and execution environment.
* **Weights & Biases (W&B)** – used for experiment tracking, visualization, and comparison of training runs.
* **Git / GitHub** – used for version control and project development.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <https://github.com/Breath-Classification/Breath-Project>
cd <Breath-Project>
```

The project uses Docker to provide the required environment and dependencies. Make sure that Docker and Docker Compose are installed on your system.

Build and start the Docker environment using:

```bash
docker compose up --build
```

Once the container is running, install the project as a Python package:

```bash
pip install -e .
```

The Docker environment contains the dependencies required to run the machine learning framework and its models.

### Mobile Application

The new cross-platform mobile application is maintained in a separate repository. Instructions for installing and running the Android and iOS application are available in the frontend repository:

[Frontend Repository](https://github.com/maciejdrywa/BreathSenseApp-frontend)

## Running the Physiology-Aware Framework

The Physiology-Aware Framework requires only this repository. The project environment and required dependencies are provided through Docker.

After completing the installation steps, start the Docker container:

```bash
docker compose up
```

Once the container is running, the framework can be used to train and evaluate the available models.

### Main Training

The main entry point is located in:

```text
<Breath-Project/Breath_dataset_and_apps/code/brp-ml-model-main/models/Pytorch>
```

Running the main script starts the default training pipeline for the **Physiology-Aware Model**:

```bash
python main.py
```

### Scripts

Additional training and evaluation scripts are available in:

```text
<Breath-Project/Breath_dataset_and_apps/code/brp-ml-model-main/scripts>
```

These scripts can be executed independently depending on the experiment or evaluation procedure being performed.

### Local Web Application

The repository also provides a simple local web application for interacting with the framework. The application allows users to:

* train and fine-tune models,
* select model and training parameters,
* experiment with different configurations,
* inspect model behaviour,
* perform model evaluation.

The application is implemented using **Streamlit**.

Because the Streamlit application runs outside the main Docker environment, a separate virtual environment should be created with access to the system packages.

Create the virtual environment:

```bash
python -m venv venv_streamlit --system-site-packages
```

Activate it:

```bash
source venv_streamlit/bin/activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install streamlit
```

Run the application:

```bash
streamlit run local_app.py --server.address=0.0.0.0 --server.port=8501
```

The application will then be available on port `8501`.


## Running the Personalization Framework

The Personalization Framework consists of three separate repositories that should be placed next to each other in the same directory:

```text
project/
├── Breath-Project/
├── BreathSenseApp-backend/
└── BreathSenseApp-frontend/
```

The framework uses Docker Compose to run the required backend services and their dependencies.

### Running the Framework

Navigate to the directory containing the three repositories and start the Docker Compose environment:

```bash
docker compose up --build
```

After the services have started, the personalization framework can be used in two ways.

### Mobile Application

The first option is to use the mobile application, which provides an interface for performing model personalization.

The installation and usage instructions for the Android and iOS applications are available in the **frontend repository**.

The backend configuration and server-side personalization setup are described in the **backend repository**.

### Testing Personalization Scripts

The personalization mechanisms can also be tested directly without using the mobile application.

Additional scripts for testing and evaluating the personalization process are available in:

```text
<Breath-Project/Breath_dataset_and_apps/code/brp-ml-model-main/scripts>
```

These scripts allow the personalization pipeline to be executed and tested directly from the command line.


## Data Collection and Preparation

The respiratory data used by the framework are collected using a **tensometric sensor mounted on a chest strap**. The sensor records changes in the circumference of the chest during breathing, which can then be processed into respiratory phase data suitable for machine learning models.

The following workflow can be used to prepare newly collected measurements for model training, fine-tuning, or evaluation.

### 1. Data Conversion

After collecting a measurement, the raw tensometric data should first be converted into the format used by the machine learning framework.

```bash
python scripts/convert_tens_to_model_input.py \
    --input "data/path/to/raw_measurement.txt" \
    --save-pretrained \
    --no-pseudo-labelling
```

The `--input` argument specifies the path to the raw measurement file.

The `--save-pretrained` option saves the converted data in the format required by the subsequent processing steps.

The `--no-pseudo-labelling` flag is used when pseudo-labels should **not** be generated during the conversion. This is useful when the data are intended to be manually labelled before being used for further training or evaluation.

### 2. Labelling

The converted data can then be labelled using the labelling script:

```bash
python scripts/labelling.py \
    "data/path/to/pretrained_measurement.txt"
```

This step assigns respiratory phase labels to the recorded signal.

### 3. Convert the Labelled Data

After labelling, the data should be converted once again so that the final file contains the required format for model training and evaluation.

```bash
python scripts/convert_tens_to_model_input.py \
    --input "data/path/to/labelled_measurement.txt"
```

The resulting file can then be used as an input for the machine learning framework.

### 4. Using the Prepared Data

Once the data have been processed, they can be used in several ways:

* **Model training** – the new data can be included in the training process and used with the main training pipeline.
* **Model evaluation** – the data can be used to evaluate an already trained model.
* **Fine-tuning** – the data can be used to adapt an existing model to a specific user.
* **Pseudo-labelling experiments** – the data can be processed using pseudo-labelling and used to investigate its effect on model personalization and performance.

For example, a fine-tuned model can be evaluated using:

```bash
python scripts/evaluate_fine_tuning.py \
    --base_model "data/path/to/base_model.pt" \
    --fine_tuned_model "data/path/to/fine_tuned_model.pt" \
    --labelled_file "data/path/to/labelled_sequence.txt"
```

The exact scripts and parameters can be adjusted depending on whether the collected data are intended for training, evaluation, or personalization experiments.


## Citation

If you use this project or its models in your research, please cite the associated publication:

```bibtex
@article{TODO,
  title = {TODO},
  author = {TODO},
  journal = {TODO},
  year = {TODO}
}
```
## License

This project is licensed under the MIT License. See the `LICENSE` file for the full license text.
