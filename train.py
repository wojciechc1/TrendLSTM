import torch
import torch.nn as nn
from model import LSTMModel
from data_loader import create_sequences, load_btc_data, get_all
from sklearn.preprocessing import MinMaxScaler



model = LSTMModel(input_size=1, hidden_size=50, num_layers=1, output_size=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# przykładowe dane (x_train, y_train) powinny być tensorami float
x_train, y_train, x_test, y_test, scaler = get_all()

for epoch in range(100):
    model.train()


    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


model.eval()
with torch.no_grad():
    predictions = model(x_test)


predictions_inv = scaler.inverse_transform(predictions.numpy())
y_test_inv = scaler.inverse_transform(y_test.numpy())


import matplotlib.pyplot as plt

plt.plot(y_test_inv, label="Prawdziwe")
plt.plot(predictions_inv, label="Przewidywane")
plt.legend()
plt.show()
