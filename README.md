# Intrusion Detection using LSTM

This project aims to develop an intrusion detection system using Long Short-Term Memory (LSTM) networks. Intrusion detection is a critical task in network security, where the objective is to detect malicious activities or attacks in a network.

## Dataset

The dataset used in this project is the UNSW-NB15 dataset, which is widely used for intrusion detection research. It contains various features related to network traffic, including both categorical and continuous variables.

- Training Dataset: `UNSW_NB15_training-set.csv`
- Testing Dataset: `UNSW_NB15_testing-set.csv`

## Approach

### Data Preprocessing

- The data is preprocessed to scale continuous features and encode categorical features.
- Rolling windows are created to convert the dataset into a suitable format for sequence models like LSTM.

### Model Architecture

- The LSTM model architecture consists of two LSTM layers with dropout regularization to prevent overfitting.
- The model is compiled with binary cross-entropy loss and Adam optimizer.

### Hyperparameter Tuning

- Grid search is used to tune hyperparameters such as the number of units, batch size, and epochs.
- The best performing hyperparameters are selected based on cross-validation performance.

## Files

- `main.py`: Main script to run the experiment, including data preprocessing, model training, and evaluation.
- `data_preprocessing.py`: Module for reading and preprocessing the dataset.
- `model.py`: Module containing functions to build, train, and evaluate the LSTM model.
- `hyperparameter_tuning.py`: Module for hyperparameter tuning using grid search.
- `UNSW_NB15_training-set.csv`: Training dataset.
- `UNSW_NB15_testing-set.csv`: Testing dataset.

## Usage

1. Clone the repository.
2. Ensure you have Python and necessary dependencies installed.
3. Run `main.py` to train and evaluate the LSTM model.
4. Optionally, run `hyperparameter_tuning.py` to perform hyperparameter tuning.

## Results

- Evaluation metrics such as precision, recall, F1 score, accuracy, AUC, and average precision are reported.
- Hyperparameter tuning results are displayed, showing the best performing hyperparameters.

