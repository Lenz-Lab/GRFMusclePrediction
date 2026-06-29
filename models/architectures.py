import math

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Vanilla LSTM sequence-to-sequence model."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


class LSTMAttentionModel(nn.Module):
    """LSTM with multi-head self-attention applied to the LSTM output."""
    def __init__(self, input_size, hidden_size, num_layers, num_heads,
                 output_size, lstm_dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True,
                                 dropout=lstm_dropout if num_layers > 1 else 0.0)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                               num_heads=num_heads,
                                               dropout=attn_dropout,
                                               batch_first=True)
        self.fc        = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out)


class CNNLSTMModel(nn.Module):
    """Two-layer CNN feature extractor followed by LSTM. Sequence length preserved."""
    def __init__(self, input_size, cnn_channels, hidden_size, num_layers,
                 output_size, dropout=0.0):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size,   cnn_channels,     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(cnn_channels * 2, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        cnn_out = self.cnn(x.permute(0, 2, 1))
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out)


class TransformerModel(nn.Module):
    """Transformer encoder with sinusoidal positional encoding."""
    def __init__(self, input_dim, output_dim, d_model, nhead,
                 num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.register_buffer('positional_encoding',
                             self._generate_positional_encoding(d_model, max_len=500))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                          num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_embedding(x)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.transformer_encoder(x)
        return self.output_layer(x)

    @staticmethod
    def _generate_positional_encoding(d_model, max_len=500):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


def build_model(model_name, bp, n_inputs, n_outputs, device):
    """
    Reconstruct a model from Optuna best_params.

    Args:
        model_name: one of 'lstm', 'lstm_attn', 'cnn_lstm', 'transformer'
        bp: dict of best hyperparameters from Optuna study
        n_inputs: number of input features
        n_outputs: number of output channels
        device: torch device to move model to

    Returns:
        nn.Module on the specified device
    """
    if model_name == 'lstm':
        return LSTMModel(n_inputs, bp['hidden_size'], bp['num_layers'],
                         n_outputs, bp['dropout_rate']).to(device)
    elif model_name == 'lstm_attn':
        return LSTMAttentionModel(n_inputs, bp['hidden_size'], bp['num_layers'],
                                  bp['num_heads'], n_outputs,
                                  bp['lstm_dropout'], bp['attn_dropout']).to(device)
    elif model_name == 'cnn_lstm':
        return CNNLSTMModel(n_inputs, bp['cnn_channels'], bp['hidden_size'],
                            bp['num_layers'], n_outputs, bp['dropout_rate']).to(device)
    elif model_name == 'transformer':
        return TransformerModel(n_inputs, n_outputs,
                                bp['d_model'], bp['num_heads'],
                                bp['num_encoder_layers'],
                                bp['ff_mult'] * bp['d_model'],
                                bp['dropout_rate']).to(device)
    else:
        raise ValueError(f'Unknown model name: {model_name}')