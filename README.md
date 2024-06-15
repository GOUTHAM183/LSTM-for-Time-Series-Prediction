# LSTM for Time Series Prediction

This repository contains an implementation of an LSTM model for time series prediction using PyTorch. The model is trained and tested on the airline passengers dataset to predict future values based on historical data.

## Description
This project demonstrates using Long Short-Term Memory (LSTM) networks for time series forecasting. LSTMs are particularly well-suited for this task due to their ability to capture temporal dependencies in sequential data. The model architecture consists of a single LSTM layer followed by a linear layer. The training process involves splitting the dataset into training and testing sets, and the model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function.

The code includes:
- Data preprocessing and train-test split
- Creation of dataset windows for the LSTM
- Definition of the LSTM model architecture
- Training loop with periodic evaluation
- Visualization of the results with matplotlib

## Dependencies
- matplotlib
- numpy
- pandas
- torch

## Usage
1. Ensure you have the required dependencies installed.
2. Download the `airline-passengers.csv` dataset and place it in the same directory as the script.
3. Run the script to train the model and visualize the results.

```bash
python lstm_time_series.py
```

## Results
The results are visualized by plotting the actual time series data along with the model's predictions on both the training and testing sets, providing a clear view of the model's performance.
