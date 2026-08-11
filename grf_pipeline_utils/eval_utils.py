import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


def _ranges(y):
    return y.max(axis=(0, 1)) - y.min(axis=(0, 1))   # (n_outputs,)


def calc_r2_per_output(y_true, y_pred, labels=None, verbose=True):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=(0, 1))
    ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))
    r2 = 1 - ss_res / ss_tot
    if verbose and labels is not None:
        for lbl, v in zip(labels, r2):
            print(f'  {lbl}: {v:.4f}')
    return r2


def calc_r2_overall(y_true, y_pred):
    return r2_score(y_true.flatten(), y_pred.flatten())


def calc_rmse_per_output(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))


def calc_rmse_overall(y_true, y_pred):
    return np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2))


def calc_rrmse_per_output(y_true, y_pred, labels=None, verbose=True):
    rrmse = calc_rmse_per_output(y_true, y_pred) / _ranges(y_true)
    if verbose and labels is not None:
        for lbl, v in zip(labels, rrmse):
            print(f'  {lbl}: {v:.4f}')
    return rrmse


def calc_rrmse_overall(y_true, y_pred):
    flat_t, flat_p = y_true.flatten(), y_pred.flatten()
    return np.sqrt(np.mean((flat_t - flat_p) ** 2)) / (flat_t.max() - flat_t.min())


def calc_rrmse_weighted(y_true, y_pred):
    r = _ranges(y_true)
    rrmse = calc_rmse_per_output(y_true, y_pred) / r
    return np.sum(rrmse * r) / np.sum(r)


def calc_rmspe_overall(y_true, y_pred):
    flat_t, flat_p = y_true.flatten(), y_pred.flatten()
    return np.sqrt(np.mean(((flat_t - flat_p) / flat_t) ** 2))


def calc_mae_per_output(y_true, y_pred, labels=None, verbose=True):
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))
    if verbose and labels is not None:
        for lbl, v in zip(labels, mae):
            print(f'  {lbl}: {v:.4f}')
    return mae


def calc_mae_overall(y_true, y_pred):
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))


def calc_flexor_ratio(y, output_keys,
                       ankle_keys=('achilles',),
                       hip_keys=('psoas', 'iliacus')):
    """Per-trial ratio of peak combined ankle-plantarflexor force to peak combined
    hip-flexor force. Reflects the distal->proximal load-redistribution pattern
    studied across the Y/OA cohorts.

    Default `ankle_keys` is 'achilles' alone: that output channel is already the
    derived sum of soleus + gaslat + gasmed (see data_utils.py), so including
    those muscles alongside it would double/triple-count their contribution.
    """
    ankle_idxs = [output_keys.index(k) for k in ankle_keys]
    hip_idxs = [output_keys.index(k) for k in hip_keys]

    ankle_signal = y[:, :, ankle_idxs].sum(axis=2)  # (n_samples, seq_len)
    hip_signal = y[:, :, hip_idxs].sum(axis=2)

    peak_ankle = ankle_signal.max(axis=1)  # (n_samples,)
    peak_hip = hip_signal.max(axis=1)

    return peak_ankle / peak_hip


def calc_gastroc_soleus_ratio(y, output_keys,
                               gastroc_keys=('gasmed',),
                               soleus_key='soleus',
                               reduction='mean'):
    """Per-trial gastrocnemius / (gastrocnemius + soleus) force ratio, mirroring
    the activation ratio in Uhlrich et al. 2022 (Sci Rep), Eq. 3:
    ratio = EMG_gastroc / (EMG_gastroc + EMG_soleus), averaged over the stance
    phase. That paper used EMG activation; this uses predicted/ground-truth
    muscle FORCE instead (the quantity this model actually predicts), so treat
    it as a force-based proxy for the same underlying redistribution, not a
    literal reproduction of their EMG numbers.

    Default `gastroc_keys=('gasmed',)` is medial gastrocnemius ALONE, matching
    the paper's Methods -- their Eq. 3 EMG_gastroc is medial-only, not
    medial+lateral combined; lateral gastrocnemius gets its own separate,
    unreported term used only inside their simulation constraint (their
    Eq. 7), never folded into the headline ratio or the reported 25±15% /
    17±19% statistics. This matches the ground-truth reproduction in
    notebooks/Gastroc_Soleus_Activation_Ratio.ipynb and the cross-stratum
    transfer metric in notebooks/CrossVal.ipynb -- do not reintroduce
    `gastroc_keys=('gaslat', 'gasmed')` as the default; summing both heads is
    a different, larger quantity (the denominator is gastroc + soleus, so
    summing isn't just a rescale) and would silently desync this function
    from those two notebooks again. Pass `gastroc_keys=('gasmed', 'gaslat')`
    explicitly if a combined summary measure is deliberately wanted for some
    other exploration.

    `reduction='mean'` (default) matches the paper's stance-averaged EMG
    definition; pass 'peak' for a peak-force variant instead.
    """
    gastroc_idxs = [output_keys.index(k) for k in gastroc_keys]
    soleus_idx = output_keys.index(soleus_key)

    gastroc_signal = y[:, :, gastroc_idxs].sum(axis=2)  # (n_samples, seq_len)
    soleus_signal = y[:, :, soleus_idx]                 # (n_samples, seq_len)

    if reduction == 'mean':
        gastroc_val = gastroc_signal.mean(axis=1)
        soleus_val = soleus_signal.mean(axis=1)
    elif reduction == 'peak':
        gastroc_val = gastroc_signal.max(axis=1)
        soleus_val = soleus_signal.max(axis=1)
    else:
        raise ValueError(f"reduction must be 'mean' or 'peak', got {reduction!r}")

    return gastroc_val / (gastroc_val + soleus_val)


def calc_dice_per_output(y_true, y_pred, labels=None, verbose=True):
    # Assumes non-negative curves (static-optimization forces), so the plain
    # overlap formula works without binarizing.
    overlap = np.sum(np.minimum(y_true, y_pred), axis=(0, 1))
    total = np.sum(y_true + y_pred, axis=(0, 1))
    dice = 2 * overlap / total
    if verbose and labels is not None:
        for lbl, v in zip(labels, dice):
            print(f'  {lbl}: {v:.4f}')
    return dice


def calc_dice_overall(y_true, y_pred):
    flat_t, flat_p = y_true.flatten(), y_pred.flatten()
    return 2 * np.sum(np.minimum(flat_t, flat_p)) / np.sum(flat_t + flat_p)


def calc_dice_per_trial(y_true, y_pred):
    """Unreduced per-trial, per-output Dice — used to feed compare_models_wilcoxon."""
    overlap = np.sum(np.minimum(y_true, y_pred), axis=1)
    total = np.sum(y_true + y_pred, axis=1)
    return 2 * overlap / total  # (n_samples, n_outputs)


def compare_models_wilcoxon(metric_a, metric_b, alternative='two-sided'):
    """Paired Wilcoxon signed-rank test between two per-trial metric arrays."""
    from scipy.stats import wilcoxon
    statistic, p_value = wilcoxon(metric_a, metric_b, alternative=alternative)
    return {'statistic': float(statistic), 'p_value': float(p_value)}


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    return model

def load_muscle_stats(filepath):
    """
    Load muscle statistics from a text file formatted like:

    Tibialis Posterior:
     Mean Max = 352.28
     Std Max = 153.86
     ...
    """

    stats = {}
    current_muscle = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # New muscle section ends with ':'
            if line.endswith(":"):
                current_muscle = line[:-1]
                stats[current_muscle] = {}
                continue

            # Parse "Key = Value"
            if "=" in line:
                key, value = line.split("=")
                key = key.strip()
                value = float(value.strip())
                stats[current_muscle][key] = value

    return stats


def eval_model(model, X_test_tensor, y_test_tensor):
    model.eval()

    test_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

        test_loss = criterion(y_pred_tensor, y_test_tensor).item()

    return test_loss, y_pred_tensor



def generate_latex_table(results_muscle_dict, results_overall_dict, muscle_labels):
    table = "\\begin{table}\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{lcccc}\n"
    table += "\\toprule\n"
    table += "\\textbf{Muscle} & \\textbf{LSTM} & \\textbf{CNN-LSTM} & \\textbf{LSTM+Attention} & \\textbf{Transformer}\\\\\n" # noqa: E501
    table += "\\midrule\n"

    for muscle, metrics in zip(muscle_labels, zip(*results_muscle_dict.values())):
        table += f"{{{muscle}}} & {metrics[0]:.4f} & {metrics[1]:.4f} & {metrics[2]:.4f} & {metrics[3]:.4f} \\\\\n" # noqa: E501

    table += "\\midrule\n"

    table += f"Overall & {results_overall_dict['LSTM']:.4f} & {results_overall_dict['CNN-LSTM']:.4f} & {results_overall_dict['LSTM+Attention']:.4f} & {results_overall_dict['Transformer']:.4f} \\\\\n" # noqa: E501

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Caption}\n"
    table += "\\label{tab:results}\n"
    table += "\\end{table}\n"

    return table
