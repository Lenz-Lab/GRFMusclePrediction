import numpy as np
import torch
import torch.nn as nn


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

def calc_r2_muscle(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=(0, 1))
    ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))

    r2_muscle = 1 - (ss_res / ss_tot)

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, r2 in zip(muscle_labels, r2_muscle):
        print(f"{label}: {r2:.4f}")

    return r2_muscle


def calc_r2_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2_overall = 1 - (ss_res / ss_tot)

    return r2_overall


def calc_rmse_muscle(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rmse_val in zip(muscle_labels, rmse):
        print(f"{label}: {rmse_val:.4f}")

    return rmse


def calc_rrmse_muscle(y_true, y_pred):
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    relative_rmse_range = rmse / ranges

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rel_range in zip(muscle_labels, relative_rmse_range):
        print(f"{label}: {rel_range:.4f}")

    return relative_rmse_range


def calc_nrmse_muscle(y_true, y_pred):
    means = y_true.mean(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    relative_rmse_mean = rmse / means

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rel_range, rel_mean in zip(muscle_labels, relative_rmse_mean):
        print(f"{label}: {rel_range:.4f}, {rel_mean:.4f}")

    return relative_rmse_mean


def calc_rmspe_muscle(y_true, y_pred):
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2, axis=(0, 1)))

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, rmspe_val in zip(muscle_labels, rmspe):
        print(f"{label}: {rmspe_val:.4f}")

    return rmspe


def calc_rmse_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmse_overall = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse_overall


def calc_rrmse_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ranges = y_true.max() - y_true.min()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    relative_rmse_overall = rmse / ranges

    return relative_rmse_overall


def calc_rrmse_weighted(y_true, y_pred):
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 1)))
    relative_rmse_range = rmse / ranges

    relative_rmse_weighted = np.sum(relative_rmse_range * ranges) / np.sum(ranges)

    return relative_rmse_weighted


def calc_rmspe_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    rmspe_overall = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

    return rmspe_overall


def calc_mae_muscle(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))  # Calculate MAE for each muscle

    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, mae_val in zip(muscle_labels, mae):
        print(f"{label}: {mae_val:.4f}")

    return mae


def calc_mae_weighted(y_true, y_pred):
    ranges = y_true.max(axis=(0, 1)) - y_true.min(axis=(0, 1))
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))  # Calculate MAE for each muscle

    relative_mae_range = mae / ranges
    relative_mae_weighted = np.sum(relative_mae_range * ranges) / np.sum(ranges)

    return relative_mae_weighted

def calc_mae_muscle_normalized_peak_force(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred), axis=(0, 1))

    peaks  = np.max(y_true, axis=1)
    average_peaks = np.mean(peaks, axis = 0)

    norm_mae = mae / average_peaks
    
    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

    for label, mae_val, norm_mae_val in zip(muscle_labels, mae, norm_mae):
        print(f"{label}: MAE:{mae_val:.4f}; Normalized MAE:{norm_mae_val:.4f}")
    
    norm_mae_overall = np.mean(norm_mae)

    return  mae, norm_mae, norm_mae_overall

def calc_mae_muscle_normalized_mass(y_true, y_pred):
    OA1_true = y_true[0:42]
    OA5_true = y_true[42:58]
    YA2_true = y_true[58:]   
    OA1_pred = y_pred[0:42]
    OA5_pred = y_pred[42:58]
    YA2_pred = y_pred[58:]
    OA1_mae = np.mean(np.abs(OA1_true - OA1_pred), axis=(0, 1))
    OA5_mae = np.mean(np.abs(OA5_true - OA5_pred), axis=(0, 1))
    YA2_mae = np.mean(np.abs(YA2_true - YA2_pred), axis=(0, 1))
    OA1_mae_norm = OA1_mae / 36.2872
    OA5_mae_norm = OA5_mae / 29.93694
    YA2_mae_norm = YA2_mae / 50.71
    norm_mae = (OA1_mae_norm + OA5_mae_norm + YA2_mae_norm) / 3
    muscle_labels = ['tibpost', 'tibant', 'edl', 'ehl',
                     'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']
    for label, norm_mae_val in zip(muscle_labels, norm_mae):
        print(f"{label}: Normalized MAE:{norm_mae_val:.4f}")
    
    norm_mae_overall = np.mean(norm_mae)
    print(f'Overall MAE Normalized by bodyweight: {norm_mae_overall}')

    return norm_mae, norm_mae_overall
    

def calc_mae_overall(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae_overall = np.mean(np.abs(y_true - y_pred))  # Calculate overall MAE

    return mae_overall


def eval_model(model, X_test_tensor, y_test_tensor):
    model.eval()

    test_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

        test_loss = criterion(y_pred_tensor, y_test_tensor).item()

    return test_loss, y_pred_tensor

def eval_threshold(y_pred):
    muscles = [
        "tibpost", "tibant", "edl", "ehl", 
        "fdl", "fhl",
        "perbrev", "perlong", "achilles"
    ]
    plantars = ['tibpost', 'fhl', 'fdl', 'perbrev', 'perlong', 'achilles']
    dorsis = ['tibant', 'ehl', 'edl']
    p_idxs = [muscles.index(m) for m in plantars]
    d_idxs = [muscles.index(m) for m in dorsis]
    OA1 = y_pred[:42]
    OA5 = y_pred[42:58]
    OA1_mass = 36.2872
    OA5_mass = 29.93694

    OA1_peaks = []

    for seg in OA1:
        pf_force = seg[:, p_idxs].sum(axis = 1)
        df_force = seg[:, d_idxs].sum(axis = 1)
        peak_pf = pf_force[78]
        df_force_toe_off = df_force[78]
        norm_force = (peak_pf - df_force_toe_off) / OA1_mass
        OA1_peaks.append(norm_force)

    OA5_peaks = []

    for seg in OA5:
        pf_force = seg[:, p_idxs].sum(axis = 1)
        df_force = seg[:, d_idxs].sum(axis = 1)
        peak_pf = pf_force[75]
        df_force_toe_off = df_force[75]
        norm_force = (peak_pf - df_force_toe_off) / OA5_mass
        OA5_peaks.append(norm_force)

    return OA1_peaks, OA5_peaks




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
