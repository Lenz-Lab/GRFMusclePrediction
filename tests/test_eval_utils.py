import numpy as np
import pytest
from grf_pipeline_utils.eval_utils import (
    calc_r2_per_output,
    calc_r2_overall,
    calc_rmse_per_output,
    calc_rmse_overall,
    calc_rrmse_per_output,
    calc_rrmse_overall,
    calc_rrmse_weighted,
    calc_mae_per_output,
    calc_mae_overall,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _perfect_preds(shape=(50, 100, 6)):
    """y_true and y_pred are identical — all metrics should be perfect."""
    y = np.random.rand(*shape)
    return y, y.copy()


def _zero_preds(shape=(50, 100, 6)):
    """y_pred is all zeros."""
    y_true = np.random.rand(*shape) + 0.1  # ensure non-zero
    y_pred = np.zeros_like(y_true)
    return y_true, y_pred


def _known_error():
    """
    y_true = [[1, 2], [3, 4]] over 2 timesteps, 2 outputs, 1 segment.
    y_pred = [[2, 2], [4, 4]] — error of 1 in output 0, 0 in output 1.
    Makes expected values easy to compute by hand.
    """
    y_true = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    y_pred = np.array([[[2.0, 2.0], [4.0, 4.0]]])  # (1, 2, 2)
    return y_true, y_pred


# ── calc_r2_per_output ─────────────────────────────────────────────────────────

def test_r2_perfect_predictions():
    """Perfect predictions should give R² = 1 for all outputs."""
    y_true, y_pred = _perfect_preds()
    r2 = calc_r2_per_output(y_true, y_pred, verbose=False)
    assert np.allclose(r2, 1.0, atol=1e-6)


def test_r2_output_shape():
    """R² should return one value per output channel."""
    y_true, y_pred = _perfect_preds(shape=(50, 100, 6))
    r2 = calc_r2_per_output(y_true, y_pred, verbose=False)
    assert r2.shape == (6,)


def test_r2_worse_than_mean_is_negative():
    """Predictions worse than predicting the mean should give R² < 0."""
    y_true = np.ones((20, 100, 3)) * 5.0
    y_true += np.random.rand(20, 100, 3) * 0.1
    # predictions far from true values
    y_pred = y_true + 10.0
    r2 = calc_r2_per_output(y_true, y_pred, verbose=False)
    assert np.all(r2 < 0)


def test_r2_verbose_with_labels(capsys):
    """Verbose mode with labels should print output."""
    y_true, y_pred = _perfect_preds(shape=(10, 100, 2))
    calc_r2_per_output(y_true, y_pred, labels=['muscle_a', 'muscle_b'], verbose=True)
    captured = capsys.readouterr()
    assert 'muscle_a' in captured.out
    assert 'muscle_b' in captured.out


def test_r2_verbose_without_labels_no_print(capsys):
    """Verbose mode without labels should not print anything."""
    y_true, y_pred = _perfect_preds(shape=(10, 100, 2))
    calc_r2_per_output(y_true, y_pred, labels=None, verbose=True)
    captured = capsys.readouterr()
    assert captured.out == ''


# ── calc_r2_overall ────────────────────────────────────────────────────────────

def test_r2_overall_perfect():
    """Perfect predictions should give overall R² = 1."""
    y_true, y_pred = _perfect_preds()
    assert np.isclose(calc_r2_overall(y_true, y_pred), 1.0, atol=1e-6)


def test_r2_overall_scalar():
    """Overall R² should return a scalar."""
    y_true, y_pred = _perfect_preds()
    result = calc_r2_overall(y_true, y_pred)
    assert np.isscalar(result)


# ── calc_rmse_per_output ───────────────────────────────────────────────────────

def test_rmse_per_output_perfect():
    """Perfect predictions should give RMSE = 0 for all outputs."""
    y_true, y_pred = _perfect_preds()
    rmse = calc_rmse_per_output(y_true, y_pred)
    assert np.allclose(rmse, 0.0, atol=1e-6)


def test_rmse_per_output_shape():
    """RMSE should return one value per output channel."""
    y_true, y_pred = _perfect_preds(shape=(50, 100, 8))
    rmse = calc_rmse_per_output(y_true, y_pred)
    assert rmse.shape == (8,)


def test_rmse_per_output_known_error():
    """RMSE should match hand-calculated value for known error."""
    y_true, y_pred = _known_error()
    rmse = calc_rmse_per_output(y_true, y_pred)
    # output 0: errors are [1, 1], RMSE = 1.0
    # output 1: errors are [0, 0], RMSE = 0.0
    assert np.isclose(rmse[0], 1.0, atol=1e-6)
    assert np.isclose(rmse[1], 0.0, atol=1e-6)


def test_rmse_per_output_non_negative():
    """RMSE should always be non-negative."""
    y_true, y_pred = _zero_preds()
    rmse = calc_rmse_per_output(y_true, y_pred)
    assert np.all(rmse >= 0)


# ── calc_rmse_overall ──────────────────────────────────────────────────────────

def test_rmse_overall_perfect():
    """Perfect predictions should give overall RMSE = 0."""
    y_true, y_pred = _perfect_preds()
    assert np.isclose(calc_rmse_overall(y_true, y_pred), 0.0, atol=1e-6)


def test_rmse_overall_scalar():
    """Overall RMSE should return a scalar."""
    y_true, y_pred = _perfect_preds()
    assert np.isscalar(calc_rmse_overall(y_true, y_pred))


def test_rmse_overall_non_negative():
    """Overall RMSE should always be non-negative."""
    y_true, y_pred = _zero_preds()
    assert calc_rmse_overall(y_true, y_pred) >= 0


# ── calc_rrmse_per_output ──────────────────────────────────────────────────────

def test_rrmse_per_output_perfect():
    """Perfect predictions should give RRMSE = 0."""
    y_true, y_pred = _perfect_preds()
    rrmse = calc_rrmse_per_output(y_true, y_pred, verbose=False)
    assert np.allclose(rrmse, 0.0, atol=1e-6)


def test_rrmse_per_output_shape():
    """RRMSE should return one value per output channel."""
    y_true, y_pred = _perfect_preds(shape=(50, 100, 6))
    rrmse = calc_rrmse_per_output(y_true, y_pred, verbose=False)
    assert rrmse.shape == (6,)


def test_rrmse_per_output_non_negative():
    """RRMSE should always be non-negative."""
    y_true, y_pred = _zero_preds()
    rrmse = calc_rrmse_per_output(y_true, y_pred, verbose=False)
    assert np.all(rrmse >= 0)


# ── calc_rrmse_overall ─────────────────────────────────────────────────────────

def test_rrmse_overall_perfect():
    """Perfect predictions should give overall RRMSE = 0."""
    y_true, y_pred = _perfect_preds()
    assert np.isclose(calc_rrmse_overall(y_true, y_pred), 0.0, atol=1e-6)


def test_rrmse_overall_scalar():
    """Overall RRMSE should return a scalar."""
    y_true, y_pred = _perfect_preds()
    assert np.isscalar(calc_rrmse_overall(y_true, y_pred))


# ── calc_rrmse_weighted ────────────────────────────────────────────────────────

def test_rrmse_weighted_perfect():
    """Perfect predictions should give weighted RRMSE = 0."""
    y_true, y_pred = _perfect_preds()
    assert np.isclose(calc_rrmse_weighted(y_true, y_pred), 0.0, atol=1e-6)


def test_rrmse_weighted_scalar():
    """Weighted RRMSE should return a scalar."""
    y_true, y_pred = _perfect_preds()
    assert np.isscalar(calc_rrmse_weighted(y_true, y_pred))


def test_rrmse_weighted_non_negative():
    """Weighted RRMSE should always be non-negative."""
    y_true, y_pred = _zero_preds()
    assert calc_rrmse_weighted(y_true, y_pred) >= 0


# ── calc_mae_per_output ────────────────────────────────────────────────────────

def test_mae_per_output_perfect():
    """Perfect predictions should give MAE = 0."""
    y_true, y_pred = _perfect_preds()
    mae = calc_mae_per_output(y_true, y_pred, verbose=False)
    assert np.allclose(mae, 0.0, atol=1e-6)


def test_mae_per_output_shape():
    """MAE should return one value per output channel."""
    y_true, y_pred = _perfect_preds(shape=(50, 100, 6))
    mae = calc_mae_per_output(y_true, y_pred, verbose=False)
    assert mae.shape == (6,)


def test_mae_per_output_known_error():
    """MAE should match hand-calculated value for known error."""
    y_true, y_pred = _known_error()
    mae = calc_mae_per_output(y_true, y_pred, verbose=False)
    # output 0: errors are [1, 1], MAE = 1.0
    # output 1: errors are [0, 0], MAE = 0.0
    assert np.isclose(mae[0], 1.0, atol=1e-6)
    assert np.isclose(mae[1], 0.0, atol=1e-6)


def test_mae_per_output_non_negative():
    """MAE should always be non-negative."""
    y_true, y_pred = _zero_preds()
    mae = calc_mae_per_output(y_true, y_pred, verbose=False)
    assert np.all(mae >= 0)


# ── calc_mae_overall ───────────────────────────────────────────────────────────

def test_mae_overall_perfect():
    """Perfect predictions should give overall MAE = 0."""
    y_true, y_pred = _perfect_preds()
    assert np.isclose(calc_mae_overall(y_true, y_pred), 0.0, atol=1e-6)


def test_mae_overall_scalar():
    """Overall MAE should return a scalar."""
    y_true, y_pred = _perfect_preds()
    assert np.isscalar(calc_mae_overall(y_true, y_pred))


def test_mae_overall_non_negative():
    """Overall MAE should always be non-negative."""
    y_true, y_pred = _zero_preds()
    assert calc_mae_overall(y_true, y_pred) >= 0


def test_mae_overall_known_error():
    """MAE overall should match hand-calculated value."""
    y_true, y_pred = _known_error()
    # total errors: [1, 0, 1, 0] flattened → mean = 0.5
    assert np.isclose(calc_mae_overall(y_true, y_pred), 0.5, atol=1e-6)
