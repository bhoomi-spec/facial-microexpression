import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]