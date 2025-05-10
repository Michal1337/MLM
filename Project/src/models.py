import math
import torch
import torch.nn as nn

class TimeSeriesMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        '''
        Feedforward MLP for time series. input_size = window length.
        hidden_sizes = list of hidden layer widths.
        '''
        super(TimeSeriesMLP, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten multi-dimensional input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
    
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rnn_type='LSTM'):
        '''
        Recurrent model (LSTM or GRU) for time series.
        input_size = feature dimension per time step.
        hidden_size = dimension of hidden state.
        '''
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = None
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, self.hidden = self.rnn(x, self.hidden)
        # Use the last output for prediction
        out = out[:, -1, :]
        return self.fc(out)
    
    def reset_hidden(self, batch_size):
        # Reinitialize hidden state (and cell state for LSTM) to zeros
        if self.rnn_type == 'lstm':
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            self.hidden = (h, c)
        else:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            self.hidden = h

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size):
        '''
        Simplified Transformer encoder for time series.
        input_size = features per step (often 1).
        d_model = embedding dimension, nhead = attention heads.
        '''
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # use last time step
        return self.fc(x)
