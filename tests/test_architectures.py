import pytest
import torch

from models.architectures import (
    CNNLSTMModel,
    LSTMAttentionModel,
    LSTMModel,
    TransformerModel,
    build_model,
)

# ── Constants ──────────────────────────────────────────────────────────────────
# Matches real pipeline dimensions: 6 GRF/COP inputs, 44 muscle/JRF outputs
BATCH   = 8
SEQ_LEN = 100
N_IN    = 6
N_OUT   = 44


def _x():
    """Synthetic input batch: (batch, seq_len, n_inputs)."""
    return torch.randn(BATCH, SEQ_LEN, N_IN)


# ── Output shape tests ─────────────────────────────────────────────────────────

def test_lstm_output_shape():
    """LSTM output should be (batch, seq_len, n_outputs)."""
    model = LSTMModel(N_IN, hidden_size=64, num_layers=2,
                      output_size=N_OUT, dropout=0.1)
    out = model(_x())
    assert out.shape == (BATCH, SEQ_LEN, N_OUT)


def test_lstm_attn_output_shape():
    """LSTM+Attention output should be (batch, seq_len, n_outputs)."""
    model = LSTMAttentionModel(N_IN, hidden_size=64, num_layers=2,
                               num_heads=4, output_size=N_OUT,
                               lstm_dropout=0.1, attn_dropout=0.1)
    out = model(_x())
    assert out.shape == (BATCH, SEQ_LEN, N_OUT)


def test_cnn_lstm_output_shape():
    """CNN-LSTM output should be (batch, seq_len, n_outputs)."""
    model = CNNLSTMModel(N_IN, cnn_channels=32, hidden_size=64,
                         num_layers=2, output_size=N_OUT, dropout=0.1)
    out = model(_x())
    assert out.shape == (BATCH, SEQ_LEN, N_OUT)


def test_transformer_output_shape():
    """Transformer output should be (batch, seq_len, n_outputs)."""
    model = TransformerModel(N_IN, N_OUT, d_model=64, nhead=4,
                             num_encoder_layers=2, dim_feedforward=128,
                             dropout=0.1)
    out = model(_x())
    assert out.shape == (BATCH, SEQ_LEN, N_OUT)


# ── NaN tests ──────────────────────────────────────────────────────────────────

def test_lstm_no_nan():
    """LSTM forward pass should not produce NaN values."""
    model = LSTMModel(N_IN, hidden_size=64, num_layers=2,
                      output_size=N_OUT, dropout=0.0)
    assert not torch.isnan(model(_x())).any()


def test_lstm_attn_no_nan():
    """LSTM+Attention forward pass should not produce NaN values."""
    model = LSTMAttentionModel(N_IN, hidden_size=64, num_layers=2,
                               num_heads=4, output_size=N_OUT,
                               lstm_dropout=0.0, attn_dropout=0.0)
    assert not torch.isnan(model(_x())).any()


def test_cnn_lstm_no_nan():
    """CNN-LSTM forward pass should not produce NaN values."""
    model = CNNLSTMModel(N_IN, cnn_channels=32, hidden_size=64,
                         num_layers=2, output_size=N_OUT, dropout=0.0)
    assert not torch.isnan(model(_x())).any()


def test_transformer_no_nan():
    """Transformer forward pass should not produce NaN values."""
    model = TransformerModel(N_IN, N_OUT, d_model=64, nhead=4,
                             num_encoder_layers=2, dim_feedforward=128,
                             dropout=0.0)
    assert not torch.isnan(model(_x())).any()


# ── build_model tests ──────────────────────────────────────────────────────────

def test_build_model_lstm():
    """build_model should correctly instantiate an LSTM from best_params."""
    bp = {'hidden_size': 64, 'num_layers': 2, 'dropout_rate': 0.1}
    model = build_model('lstm', bp, N_IN, N_OUT, torch.device('cpu'))
    assert model(_x()).shape == (BATCH, SEQ_LEN, N_OUT)


def test_build_model_lstm_attn():
    """build_model should correctly instantiate an LSTM+Attention model."""
    bp = {'hidden_size': 64, 'num_layers': 2, 'num_heads': 4,
          'lstm_dropout': 0.1, 'attn_dropout': 0.1}
    model = build_model('lstm_attn', bp, N_IN, N_OUT, torch.device('cpu'))
    assert model(_x()).shape == (BATCH, SEQ_LEN, N_OUT)


def test_build_model_cnn_lstm():
    """build_model should correctly instantiate a CNN-LSTM model."""
    bp = {'cnn_channels': 32, 'hidden_size': 64,
          'num_layers': 2, 'dropout_rate': 0.1}
    model = build_model('cnn_lstm', bp, N_IN, N_OUT, torch.device('cpu'))
    assert model(_x()).shape == (BATCH, SEQ_LEN, N_OUT)


def test_build_model_transformer():
    """build_model should correctly instantiate a Transformer model."""
    bp = {'d_model': 64, 'num_heads': 4, 'num_encoder_layers': 2,
          'ff_mult': 2, 'dropout_rate': 0.1}
    model = build_model('transformer', bp, N_IN, N_OUT, torch.device('cpu'))
    assert model(_x()).shape == (BATCH, SEQ_LEN, N_OUT)


def test_build_model_unknown_raises():
    """build_model should raise ValueError for an unknown model name."""
    bp = {}
    with pytest.raises(ValueError):
        build_model('unknown_model', bp, N_IN, N_OUT, torch.device('cpu'))
