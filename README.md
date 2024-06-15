LSTM for Time Series Prediction
This repository contains an implementation of an LSTM model for time series prediction using PyTorch. The model is trained and tested on the airline passengers dataset to predict future values based on historical data.

Introduction
This project demonstrates the use of Long Short-Term Memory (LSTM) networks for time series forecasting. LSTMs are particularly well-suited for this task due to their ability to capture temporal dependencies in sequential data.

Dependencies
The following libraries are required to run the code:

matplotlib
numpy
pandas
torch
You can install these dependencies using pip:

bash
Copy code
pip install matplotlib numpy pandas torch
Dataset
The dataset used in this project is the airline passengers dataset, which contains monthly totals of international airline passengers from 1949 to 1960. The dataset is available as airline-passengers.csv.

Model Architecture
The model architecture consists of a single LSTM layer followed by a linear layer. The LSTM layer has 50 hidden units.

python
Copy code
class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
Training the Model
The model is trained for 2000 epochs using the Adam optimizer and Mean Squared Error (MSE) loss function. The training data is split into training and testing sets with a 67% - 33% ratio.

python
Copy code
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
Evaluation
The model's performance is evaluated using Root Mean Squared Error (RMSE) on both the training and testing sets. The predictions are plotted alongside the actual values to visually assess the model's performance.

Results
The results are plotted showing the actual time series data, the training predictions, and the testing predictions.

python
Copy code
plt.plot(timeseries)
plt.plot(train_plot, c='r')
plt.plot(test_plot, c='g')
plt.show()
Usage
To run the code, follow these steps:

Ensure you have the required dependencies installed.
Download the airline-passengers.csv dataset and place it in the same directory as the script.
Run the script to train the model and visualize the results.
bash
Copy code
python lstm_time_series.py
