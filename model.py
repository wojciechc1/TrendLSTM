import torch.nn as nn
import torch


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # warstwa LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # warstwa liniowa na wyjście
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # output LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_len, hidden_size)

        # bierzemy wyjście z ostatniego kroku czasowego
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # przez warstwę liniową
        out = self.fc(out)  # (batch_size, output_size)
        return out
