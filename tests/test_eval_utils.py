import inspect

import numpy as np
import pytest

from grf_pipeline_utils.eval_utils import (
    calc_dice_overall,
    calc_dice_per_output,
    calc_dice_per_trial,
    calc_flexor_ratio,
    calc_gastroc_soleus_ratio,
    calc_mae_overall,
    calc_mae_per_output,
    calc_r2_overall,
    calc_r2_per_output,
    calc_rmse_overall,
    calc_rmse_per_output,
    calc_rrmse_overall,
    calc_rrmse_per_output,
    calc_rrmse_weighted,
    compare_models_wilcoxon,
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


# ── calc_flexor_ratio ──────────────────────────────────────────────────────────

_FLEXOR_KEYS = ['soleus', 'gaslat', 'gasmed', 'achilles', 'psoas', 'iliacus']


def _flexor_fixture():
    """
    2 samples, 3 timesteps, 6 outputs ordered per _FLEXOR_KEYS.
    Sample 0: ankle group (soleus only) peaks at 3, hip group (psoas only) peaks at 1 -> ratio 3.0
    Sample 1: ankle group peaks at 2, hip group (psoas only) peaks at 4 -> ratio 0.5
    """
    y = np.zeros((2, 3, 6))
    y[0, :, 0] = [1, 2, 3]   # soleus
    y[0, :, 4] = [1, 1, 1]   # psoas
    y[1, :, 0] = [2, 2, 2]   # soleus
    y[1, :, 4] = [4, 4, 4]   # psoas
    return y


def test_flexor_ratio_known_values():
    """Ratio should match hand-calculated peak-ankle / peak-hip per trial."""
    y = _flexor_fixture()
    ratio = calc_flexor_ratio(y, _FLEXOR_KEYS, ankle_keys=('soleus',), hip_keys=('psoas',))
    assert np.allclose(ratio, [3.0, 0.5], atol=1e-6)


def test_flexor_ratio_shape():
    """Ratio should return one value per trial."""
    y = _flexor_fixture()
    ratio = calc_flexor_ratio(y, _FLEXOR_KEYS, ankle_keys=('soleus',), hip_keys=('psoas',))
    assert ratio.shape == (2,)


def test_flexor_ratio_equal_groups_is_one():
    """Identical ankle and hip group magnitudes should give ratio 1.0."""
    y = np.zeros((1, 3, 6))
    y[0, :, 0] = [1, 2, 3]   # soleus (ankle group)
    y[0, :, 4] = [1, 2, 3]   # psoas (hip group)
    ratio = calc_flexor_ratio(y, _FLEXOR_KEYS, ankle_keys=('soleus',), hip_keys=('psoas',))
    assert np.isclose(ratio[0], 1.0, atol=1e-6)


def test_flexor_ratio_default_ankle_group_is_achilles_only():
    """
    Default ankle_keys must be ('achilles',) alone -- 'achilles' is already the
    derived sum of soleus + gaslat + gasmed (see data_utils.py), so including
    those muscles too would double/triple-count their contribution.
    """
    default_ankle_keys = inspect.signature(calc_flexor_ratio).parameters['ankle_keys'].default
    assert default_ankle_keys == ('achilles',)


def test_flexor_ratio_default_hip_group_is_psoas_iliacus():
    """Default hip_keys should be the iliopsoas pair (psoas + iliacus)."""
    default_hip_keys = inspect.signature(calc_flexor_ratio).parameters['hip_keys'].default
    assert default_hip_keys == ('psoas', 'iliacus')


# ── calc_gastroc_soleus_ratio ────────────────────────────────────────────────────

# Reuses _FLEXOR_KEYS' ordering: soleus, gaslat, gasmed, achilles, psoas, iliacus
_GASTROC_KEYS = _FLEXOR_KEYS


def test_gastroc_soleus_ratio_default_gastroc_keys_is_medial_only():
    """Default gastroc_keys should be medial gastrocnemius ALONE, matching the
    paper's Eq. 3 Methods (not medial+lateral combined) -- see eval_utils.py's
    docstring for why summing both heads would be a different quantity, not a
    rescale, and would desync this function from the two notebooks that
    reproduce the paper's ground truth."""
    default_gastroc_keys = inspect.signature(calc_gastroc_soleus_ratio).parameters['gastroc_keys'].default
    assert default_gastroc_keys == ('gasmed',)


def _gastroc_soleus_fixture():
    """
    2 samples, 3 timesteps, 6 outputs ordered per _GASTROC_KEYS. gaslat is
    deliberately left nonzero to prove the medial-only default ignores it.
    Sample 0: gasmed=2 (constant), gaslat=99 (constant, must be ignored);
              soleus=4 (constant) -> mean ratio = 2/(2+4) = 0.3333...
    Sample 1: gasmed=1 (constant), gaslat=99 (constant, must be ignored);
              soleus=6 (constant) -> mean ratio = 1/(1+6) = 0.142857...
    """
    y = np.zeros((2, 3, 6))
    y[0, :, 1] = [99, 99, 99]   # gaslat -- must not affect the default-key result
    y[0, :, 2] = [2, 2, 2]      # gasmed
    y[0, :, 0] = [4, 4, 4]      # soleus
    y[1, :, 1] = [99, 99, 99]   # gaslat -- must not affect the default-key result
    y[1, :, 2] = [1, 1, 1]      # gasmed
    y[1, :, 0] = [6, 6, 6]      # soleus
    return y


def test_gastroc_soleus_ratio_known_values_mean():
    """Default (mean) reduction should match hand-calculated gasmed/(gasmed+soleus),
    ignoring gaslat entirely."""
    y = _gastroc_soleus_fixture()
    ratio = calc_gastroc_soleus_ratio(y, _GASTROC_KEYS)
    assert np.allclose(ratio, [2 / 6, 1 / 7], atol=1e-6)


def test_gastroc_soleus_ratio_shape():
    """Ratio should return one value per trial."""
    y = _gastroc_soleus_fixture()
    ratio = calc_gastroc_soleus_ratio(y, _GASTROC_KEYS)
    assert ratio.shape == (2,)


def test_gastroc_soleus_ratio_peak_reduction():
    """reduction='peak' should use each signal's peak instead of its stance-mean."""
    y = np.zeros((1, 3, 6))
    y[0, :, 2] = [1, 5, 1]   # gasmed, peak=5
    y[0, :, 1] = [9, 9, 9]   # gaslat, must be ignored by the medial-only default
    y[0, :, 0] = [2, 2, 2]   # soleus, peak=2
    ratio = calc_gastroc_soleus_ratio(y, _GASTROC_KEYS, reduction='peak')
    assert np.isclose(ratio[0], 5 / 7, atol=1e-6)


def test_gastroc_soleus_ratio_equal_signals_is_half():
    """Equal gastroc and soleus signals should give ratio 0.5."""
    y = np.zeros((1, 3, 6))
    y[0, :, 2] = [3, 3, 3]   # gasmed
    y[0, :, 1] = [7, 7, 7]   # gaslat, must be ignored by the medial-only default
    y[0, :, 0] = [3, 3, 3]   # soleus
    ratio = calc_gastroc_soleus_ratio(y, _GASTROC_KEYS)
    assert np.isclose(ratio[0], 0.5, atol=1e-6)


def test_gastroc_soleus_ratio_invalid_reduction_raises():
    """An unrecognized reduction mode should raise ValueError."""
    y = _gastroc_soleus_fixture()
    with pytest.raises(ValueError):
        calc_gastroc_soleus_ratio(y, _GASTROC_KEYS, reduction='bogus')


def test_gastroc_soleus_ratio_explicit_combined_heads():
    """Explicitly opting into gastroc_keys=('gaslat', 'gasmed') should sum both
    heads -- the old default, still supported, just no longer automatic."""
    y = _gastroc_soleus_fixture()
    ratio = calc_gastroc_soleus_ratio(y, _GASTROC_KEYS, gastroc_keys=('gaslat', 'gasmed'))
    assert np.allclose(ratio, [(99 + 2) / (99 + 2 + 4), (99 + 1) / (99 + 1 + 6)], atol=1e-6)


# ── calc_dice_per_output ───────────────────────────────────────────────────────

def test_dice_per_output_perfect():
    """Perfect predictions should give Dice = 1 for all outputs."""
    y_true, y_pred = _perfect_preds()
    dice = calc_dice_per_output(y_true, y_pred, verbose=False)
    assert np.allclose(dice, 1.0, atol=1e-6)


def test_dice_per_output_shape():
    """Dice should return one value per output channel."""
    y_true, y_pred = _perfect_preds(shape=(50, 100, 6))
    dice = calc_dice_per_output(y_true, y_pred, verbose=False)
    assert dice.shape == (6,)


def test_dice_per_output_known_values():
    """Dice should match hand-calculated value for known non-negative curves."""
    y_true, y_pred = _known_error()
    dice = calc_dice_per_output(y_true, y_pred, verbose=False)
    # output 0: true=[1,3], pred=[2,4] -> 2*(1+3)/((1+2)+(3+4)) = 8/10 = 0.8
    # output 1: true=[2,4], pred=[2,4] -> 2*(2+4)/((2+2)+(4+4)) = 12/12 = 1.0
    assert np.isclose(dice[0], 0.8, atol=1e-6)
    assert np.isclose(dice[1], 1.0, atol=1e-6)


def test_dice_per_output_zero_preds_is_zero():
    """All-zero predictions against positive ground truth should give Dice = 0."""
    y_true, y_pred = _zero_preds()
    dice = calc_dice_per_output(y_true, y_pred, verbose=False)
    assert np.allclose(dice, 0.0, atol=1e-6)


# ── calc_dice_overall ──────────────────────────────────────────────────────────

def test_dice_overall_perfect():
    """Perfect predictions should give overall Dice = 1."""
    y_true, y_pred = _perfect_preds()
    assert np.isclose(calc_dice_overall(y_true, y_pred), 1.0, atol=1e-6)


def test_dice_overall_known_value():
    """Overall Dice should match hand-calculated value."""
    y_true, y_pred = _known_error()
    # flattened: 2*(1+2+3+4)/((1+2)+(2+2)+(3+4)+(4+4)) = 20/22
    assert np.isclose(calc_dice_overall(y_true, y_pred), 20 / 22, atol=1e-6)


# ── calc_dice_per_trial ────────────────────────────────────────────────────────

def test_dice_per_trial_shape():
    """Per-trial Dice should preserve the sample and output dimensions."""
    y_true, y_pred = _perfect_preds(shape=(50, 100, 6))
    dice = calc_dice_per_trial(y_true, y_pred)
    assert dice.shape == (50, 6)


def test_dice_per_trial_known_values():
    """Per-trial Dice should match hand-calculated per-output values for a single trial."""
    y_true, y_pred = _known_error()
    dice = calc_dice_per_trial(y_true, y_pred)
    assert np.allclose(dice, [[0.8, 1.0]], atol=1e-6)


# ── compare_models_wilcoxon ────────────────────────────────────────────────────

def test_wilcoxon_returns_dict_with_expected_keys():
    """Result should be a dict with 'statistic' and 'p_value' keys."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=20)
    b = rng.normal(size=20)
    result = compare_models_wilcoxon(a, b)
    assert set(result.keys()) == {'statistic', 'p_value'}


def test_wilcoxon_near_identical_arrays_high_pvalue():
    """Paired samples with only tiny, non-systematic noise shouldn't be flagged as significant."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = a + rng.normal(0, 0.01, 30)
    result = compare_models_wilcoxon(a, b)
    assert result['p_value'] > 0.05


def test_wilcoxon_shifted_arrays_low_pvalue():
    """A systematic shift between paired samples should be detected as significant."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = a + 5.0
    result = compare_models_wilcoxon(a, b)
    assert result['p_value'] < 0.05
