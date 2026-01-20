import torch 
import torch.nn as nn
import torch.optim as optim
import time


#Function to define the ML model
class LSTMModel(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else architecture[i - 1], hidden_size, batch_first=True))
        self.fc = nn.Linear(architecture[-1], 2)
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    

class LSTM_EIR(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTM_EIR, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else architecture[i - 1], hidden_size, batch_first=True))
        self.fc = nn.Linear(architecture[-1], 1)  # Predicting only EIR
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class LSTM_Incidence(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTM_Incidence, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else architecture[i - 1], hidden_size, batch_first=True))
        self.fc = nn.Linear(architecture[-1], 1)  # Predicting only Incidence
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    
class LSTMModel_baseline(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTMModel_baseline, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size if i == 0 else architecture[i - 1],
                    hidden_size,
                    batch_first=True
                )
            )
        self.fc = nn.Linear(architecture[-1] + 1, 2)  # output 2 values

    def forward(self, x_seq, x_scalar):
        x = x_seq
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        last_hidden = x[:, -1, :]  # (batch_size, hidden_size)
        combined = torch.cat((last_hidden, x_scalar), dim=1)  # (batch_size, hidden_size + 1)
        return self.fc(combined)  # (batch_size, 2)
    
class MultiScaleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, k, dilation=d,
                          padding=((k - 1) // 2) * d),
                nn.ReLU(),
                nn.BatchNorm1d(16)
            )
            for k, d in zip((5, 9, 15), (1, 4, 8))
        ])

    def forward(self, x):
        return torch.cat([s(x) for s in self.scales], dim=1)
    
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, x, mask):
        scores = self.v(torch.tanh(self.q(x))).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        w = torch.softmax(scores, dim=1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)

class EIRModel(nn.Module):
    def __init__(self, architecture=[256, 128, 64,32]):
        super().__init__()
        self.conv = MultiScaleConv()
        feat_dim = 16 * 3

        self.lstm = nn.ModuleList()
        for i, h in enumerate(architecture):
            inp = feat_dim if i == 0 else architecture[i - 1]
            self.lstm.append(nn.LSTM(inp, h, batch_first=True))

        self.attn = SelfAttention(architecture[-1])
        self.head = nn.Linear(architecture[-1], 1)

    def forward(self, x, mask):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        for l in self.lstm:
            x, _ = l(x)
        x = self.attn(x, mask)
        return self.head(x)
