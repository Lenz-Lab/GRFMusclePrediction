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
    table += "\\textbf{Muscle} & \\textbf{LSTM} & \\textbf{CNN-LSTM} & \\textbf{LSTM+Attention} & \\textbf{Transformer}\\\\\n"
    table += "\\midrule\n"

    for muscle, metrics in zip(muscle_labels, zip(*results_muscle_dict.values())):
        table += f"{{{muscle}}} & {metrics[0]:.4f} & {metrics[1]:.4f} & {metrics[2]:.4f} & {metrics[3]:.4f} \\\\\n"

    table += "\\midrule\n"

    table += f"Overall & {results_overall_dict['LSTM']:.4f} & {results_overall_dict['CNN-LSTM']:.4f} & {results_overall_dict['LSTM+Attention']:.4f} & {results_overall_dict['Transformer']:.4f} \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Caption}\n"
    table += "\\label{tab:results}\n"
    table += "\\end{table}\n"

    return table
