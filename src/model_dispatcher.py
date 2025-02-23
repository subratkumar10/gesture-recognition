import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

import config

class CustomMotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM_1 = nn.LSTM(input_size=config.EXTRACTED_FEATURES, hidden_size=16, num_layers=1, bidirectional=True, batch_first=True)
        self.LSTM_1_1 = nn.LSTM(input_size=2*16, hidden_size=32, bidirectional=True, batch_first=True)

        self.in_features_1 = 64
        self.LSTM_2 = nn.LSTM(input_size=2*32, hidden_size= self.in_features_1, batch_first=True)
        self.dropout_1 = nn.Dropout(0.5)

        self.linear_1 = nn.Sequential(
            nn.BatchNorm1d(num_features=self.in_features_1),
            nn.Linear(in_features=self.in_features_1, out_features=16),
            nn.Tanh()
        )
        self.linear_2 = nn.Sequential(
            nn.BatchNorm1d(num_features=16),
            nn.Linear(in_features=16, out_features=config.NUM_CLASSES)
        )

    def forward(self, x):
        out, (_, _) = self.LSTM_1(x)
        out, (_, _) = self.LSTM_1_1(out)
        _, (out, _) = self.LSTM_2(out)
        # print(type(out))
        # out = pad_packed_sequence(out, batch_first=True, padding_value=0)[0]
        out = out.reshape(out.shape[1], -1)
        out = self.dropout_1(out)
        out = self.linear_1(out)
        out = self.dropout_1(out)
        out = self.linear_2(out)
        return out

def dispatch_model(name_of_model):
    model_dict = {
        "CustomMotionModel": CustomMotionModel(),
    }
    return model_dict[name_of_model]