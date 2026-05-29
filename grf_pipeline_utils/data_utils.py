import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
from collections import Counter
import re


# define data labels
GRF_LABELS = ['GRF_x', 'GRF_y', 'GRF_z']
MUSCLE_LABELS = ['tibpost', 'tibant', 'edl', 'ehl', 'fdl', 'fhl', 'perbrev', 'perlong', 'achilles']

# define index for data labels
GRF_DICT = {0: 'GRF_x', 1: 'GRF_y', 2: 'GRF_z'}
MUSCLE_DICT = {0: 'tibpost', 1: 'tibant', 2: 'edl', 3: 'ehl', 4: 'fdl', 5: 'fhl', 6: 'perbrev', 7: 'perlong', 8: 'achilles'}


def ad2float(array_double_col):
    """
    Convert an OpenSim ArrayDouble into a Numpy array of floats

    Args:
        array_double_col (object): OpenSim object with methods '.getSize()' and '.get(i)' for retrieving values

    Returns:
        np.ndarray: 1D Numpy array of floats containing the values from 'array_double_col'
    """
    # extract each value and cast to float
    float_col = [float(array_double_col.get(i)) for i in range(array_double_col.getSize())]

    return np.array(float_col)


def interp_segments(segments: list, n_interp_points: int):
    """
    Resample segments to a fixed number of interpolation points using linear interpolation

    Args:
        segments (list): list of 1D arrays representing signal segments of varying lengths
        n_interp_points (int): number of interpolation points to resample each segment to

    Returns:
        segments_resampled (list): list of resampled segments, each with length 'n_interp_points'
        time_resampled (list): list of resampled time vectors corresponding to each segment
    """
    segments_resampled = []
    time_resampled = []

    num_segments = len(segments)

    for i in range(num_segments):
        # original segment norimalized to [0, 1] in time
        original_time = np.linspace(0, 1, len(segments[i]))
        resampled_time = np.linspace(0, 1, n_interp_points)

        # linear interpolation of the segment
        interp_func = interp1d(original_time, segments[i], kind='linear')
        interp_seg = interp_func(resampled_time)

        segments_resampled.append(interp_seg)
        time_resampled.append(resampled_time)

    return segments_resampled, time_resampled


def exclude_segments(segments: list, min_len: int, max_len: int):
    """
    Filter out segments whose length falls outside a specified range

    Args:
        segments (list): list of 1D arrays representing signal segments
        min_len (int): minimum allowable length for a segment
        max_len (int): maximum allowable length for a segment

    Returns:
        list: filtered list of segments within the length range
    """
    return [seg for seg in segments if len(seg) >= min_len and len(seg) <= max_len]


def load_data(data_dir: str):
    """
    Load ground reaction force and estimated muscle force data from .npy files and concatenate

    Args:
        data_dir (str): path to the directory containing 'grf.npy' and 'muscle.npy'

    Returns:
        np.ndarray: Combined data array with shape (samples, timesteps, channels), where channels = GRF (3) + muscle forces (n)
    """
    # load the data with memory mapping for efficiency
    grf_data = np.load(data_dir + 'grf.npy', mmap_mode='r')
    muscle_data = np.load(data_dir + 'muscle.npy', mmap_mode='r')

    # concatenate along the last axis (channel dimension)
    data = np.concatenate((grf_data, muscle_data), axis=2)

    return data


def split_data(data: np.ndarray):
    """
    Split data into training, validation, and test sets (80/10/10 split)

    Args:
        data (np.ndarray): full dataset of shape (samples, timesteps, channels)

    Returns:
        tuples: X_train, y_train, X_val, y_val, X_test, y_test arrays, where X contains GRF channels (first 3), y contains muscle force channels (9)
    """
    # ensure reproducibility
    np.random.seed(42)
    np.random.shuffle(data)

    num_samples = data.shape[0]
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    print(f"Number of samples: {num_samples}")
    # print(f"Train size: {train_size}")
    # print(f"Validation size: {val_size}")
    print("--------------------")

    # split dataset
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print("--------------------")

    # separate inputs (GRF: first 3 channels) and outputs (muscles: remaining 9)
    X_train, y_train = train_data[:, :, :3], train_data[:, :, 3:]
    X_val, y_val = val_data[:, :, :3], val_data[:, :, 3:]
    X_test, y_test = test_data[:, :, :3], test_data[:, :, 3:]

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_achilles_force(y_train: np.ndarray, muscle_index: int = 8):
    """
    Plot Achilles tendon force across stance for training samples

    Args:
        y_train (np.ndarray): training output data of shape (samples, timesteps, muscles)
        muscle_index (int): Achilles tendon is asumed to be at index 8 (last in order)

    Returns:
        fig, ax (matplotlib Figure and Axes): figure and axes objects for further customization
    """
    num_samples, num_timesteps, _ = y_train.shape
    perc_stance = np.linspace(0, 100, num_timesteps)  # percent stance (0–100%)

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot all samples with transparency
    for i in range(num_samples):
        ax.plot(perc_stance, y_train[i, :, muscle_index], linewidth=2, color="#A2C7E7", alpha=0.4)

    # overlay the mean curve in bold
    mean_curve = y_train[:, :, muscle_index].mean(axis=0)
    ax.plot(perc_stance, mean_curve, linewidth=3, color='#1A5EB6', label="mean")

    ax.set_ylabel("Muscle Force (N)", fontsize=14)
    ax.set_xlabel("Percent Normalized Stance (%)", fontsize=14)
    ax.set_title("Achilles Tendon Force Across Training Samples", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    return fig, ax

# --- Util methods written by BK for use in data batch processing ---

def data_to_segs(muscles, seg_times, problem_trials, grf_pickle_dir, muscle_force_dir,
                 problematic_seg_keys, add_achilles=True,
                 jrf_dir=None, jrf_filename='results_JointReaction_ReactionLoads.sto',
                 jrf_col_map=None):
    """
    jrf_col_map: dict mapping output key -> {'r': sto_col_right, 'l': sto_col_left}
    Example:
        jrf_col_map = {
            'subtalar_calc_fx': {'r': 'subtalar_r_on_calcn_r_in_calcn_r_fx',
                                 'l': 'subtalar_l_on_calcn_l_in_calcn_l_fx'},
            ...
        }
    When jrf_dir and jrf_col_map are both provided, JRF segments are loaded from
    {jrf_dir}/{trial_name}/{jrf_filename} and appended to each subject dict alongside
    the muscle and GRF data.
    """
    compiled_segs = {}
    fetch_jrf = (jrf_dir is not None) and (jrf_col_map is not None)

    import opensim as osim  # only needed for OpenSim data loading

    base_muscles = sorted({m[:-2] for m in muscles if m.endswith(("_r", "_l"))})

    # helper to read multiple columns from Storage dynamically
    def load_muscle_columns(storage, base_muscles, side_suffix):
        """
        Returns: (muscle_time, muscle_data) where muscle_data maps base name -> np array
        """
        muscle_time_col = osim.ArrayDouble()
        storage.getTimeColumn(muscle_time_col)
        muscle_time = ad2float(muscle_time_col)

        muscle_data = {}
        for m in base_muscles:
            col = osim.ArrayDouble()
            storage.getDataColumn(f"{m}_{side_suffix}", col)
            muscle_data[m] = ad2float(col)

        return muscle_time, muscle_data

    def load_jrf_columns(storage, jrf_col_map, side):
        """
        Returns: (jrf_time, jrf_data) where jrf_data maps output key -> np array
        for the given side ('r' or 'l').
        """
        jrf_time_col = osim.ArrayDouble()
        storage.getTimeColumn(jrf_time_col)
        jrf_time = ad2float(jrf_time_col)

        jrf_data = {}
        for out_key, col_names in jrf_col_map.items():
            col = osim.ArrayDouble()
            storage.getDataColumn(col_names[side], col)
            jrf_data[out_key] = ad2float(col)

        return jrf_time, jrf_data

    problematic_segs = []
    # loop thru all subjects and create their dictionary slots for each muscle specified
    for subject, trials in seg_times.items():
        compiled_segs[subject] = {
            'grf_x': [], 'grf_y': [], 'grf_z': [],
            'cop_x': [], 'cop_y': [], 'cop_z': [],
            **{m: [] for m in base_muscles}
        }
        if add_achilles:
            compiled_segs[subject]['achilles'] = []
        if fetch_jrf:
            for out_key in jrf_col_map:
                compiled_segs[subject][out_key] = []

        # loop thru each trial, extracting gait segments according to masks
        for trial_name, seg_dict in trials.items():
            # load grf
            grf_path = os.path.join(grf_pickle_dir, trial_name)
            grf_df = pd.read_pickle(grf_path)
            time = grf_df['time'].values
            # load muscle data
            muscle_path = os.path.join(muscle_force_dir, trial_name, 'results_forces.sto')
            muscle_storage = osim.Storage(muscle_path)
            muscle_time, muscle_r = load_muscle_columns(storage=muscle_storage, base_muscles=base_muscles, side_suffix='r')
            _, muscle_l = load_muscle_columns(storage=muscle_storage, base_muscles=base_muscles, side_suffix='l')
            # load JRF data if requested
            if fetch_jrf:
                jrf_path = os.path.join(jrf_dir, trial_name, jrf_filename)
                jrf_storage = osim.Storage(jrf_path)
                jrf_time, jrf_r = load_jrf_columns(jrf_storage, jrf_col_map, 'r')
                _, jrf_l = load_jrf_columns(jrf_storage, jrf_col_map, 'l')

            # segment loop
            for side, seg_list in seg_dict.items():
                side = side.lower()
                if side not in ("right", "left"):
                    continue
                side_char = 'r' if side == 'right' else 'l'
                for (s, e) in seg_list:
                    # skip segments flagged for bad activation values
                    key = (trial_name, side, round(s, 4), round(e, 4))
                    if key in problematic_seg_keys:
                        continue
                    grf_mask = (time >= s) & (time <= e)
                    m_mask   = (muscle_time >= s) & (muscle_time <= e)

                    if (not grf_mask.any()) or (not m_mask.any()):
                        continue

                    # pick GRF columns based on side (matching your current naming)
                    if side == "right":
                        force_seg_x = grf_df.loc[grf_mask, "ground_force_vx"].to_numpy()
                        force_seg_y = grf_df.loc[grf_mask, "ground_force_vy"].to_numpy()
                        force_seg_z = grf_df.loc[grf_mask, "ground_force_vz"].to_numpy()
                        pressure_seg_x = grf_df.loc[grf_mask, "ground_force_new_px"].to_numpy()
                        pressure_seg_y = grf_df.loc[grf_mask, "ground_force_py"].to_numpy()
                        pressure_seg_z = grf_df.loc[grf_mask, "ground_force_pz"].to_numpy()
                        mdata = muscle_r
                    else:
                        force_seg_x = grf_df.loc[grf_mask, "1_ground_force_vx"].to_numpy()
                        force_seg_y = grf_df.loc[grf_mask, "1_ground_force_vy"].to_numpy()
                        force_seg_z = (-1.0 * grf_df.loc[grf_mask, "1_ground_force_vz"]).to_numpy()
                        pressure_seg_x = grf_df.loc[grf_mask, "1_ground_force_new_px"].to_numpy()
                        pressure_seg_y = grf_df.loc[grf_mask, "1_ground_force_py"].to_numpy()
                        pressure_seg_z = grf_df.loc[grf_mask, "1_ground_force_pz"].to_numpy()
                        mdata = muscle_l

                    compiled_segs[subject]["grf_x"].append(force_seg_x)
                    compiled_segs[subject]["grf_y"].append(force_seg_y)
                    compiled_segs[subject]["grf_z"].append(force_seg_z)
                    compiled_segs[subject]["cop_x"].append(pressure_seg_x)
                    compiled_segs[subject]["cop_y"].append(pressure_seg_y)
                    compiled_segs[subject]["cop_z"].append(pressure_seg_z)

                    # muscles in a loop (the whole point)
                    seg_muscles = {}
                    for m in base_muscles:
                        seg_m = mdata[m][m_mask]
                        compiled_segs[subject][m].append(seg_m)
                        seg_muscles[m] = seg_m  # keep for achilles calc

                    # optional achilles
                    if add_achilles:
                        seg_achilles = (mdata['gaslat'][m_mask] + mdata['gasmed'][m_mask] + mdata['soleus'][m_mask])
                        compiled_segs[subject]['achilles'].append(seg_achilles)

                    # optional JRF
                    if fetch_jrf:
                        jrf_side = jrf_r if side == 'right' else jrf_l
                        jrf_mask = (jrf_time >= s) & (jrf_time <= e)
                        for out_key in jrf_col_map:
                            if side == 'left' and (out_key == 'knee_fz' or out_key == 'ankle_fz'):
                                jrf_side[out_key][jrf_mask] = -1 * jrf_side[out_key][jrf_mask] 
                            compiled_segs[subject][out_key].append(jrf_side[out_key][jrf_mask])

    return compiled_segs, problematic_segs

def get_all_segments(resampled_segs, key):
    """
    Collects all resampled segments for a given signal across all subjects.
    """
    all_segs = []
    for subj, data in resampled_segs.items():
        if subj == "time_resampled":
            continue
        if key in data:
            all_segs.extend(data[key])
    return np.array(all_segs)

def flatten_to_muscle_dict(seg_dict, muscle_keys):
    """
    Convert a subject-keyed dict  {subj: {muscle: [segs...]}, ...}
    into a muscle-keyed dict      {muscle: np.array of shape (N, T)}
    suitable for plot_muscle_grid.
    """
    out = {muscle: [] for muscle in muscle_keys}
    for subj, data in seg_dict.items():
        if not isinstance(data, dict):
            continue
        for muscle in muscle_keys:
            out[muscle].extend(data.get(muscle, []))
    return {m: np.array(segs) for m, segs in out.items() if len(segs) > 0}

def normalize_by_mass_in_order(seg_dict, all_masses, keys_to_normalize):
    out = {}

    subject_keys = [k for k, v in seg_dict.items() if isinstance(v, dict)]

    if len(subject_keys) != len(all_masses):
        raise ValueError("Mass list length does not match number of subjects")

    for subj, mass in zip(subject_keys, all_masses):
        subj_dict = seg_dict[subj]
        out[subj] = {}

        for key, seg_list in subj_dict.items():
            if not isinstance(seg_list, list):
                out[subj][key] = seg_list
                continue

            if key in keys_to_normalize:
                out[subj][key] = [np.asarray(seg) / mass for seg in seg_list]
            else:
                out[subj][key] = seg_list

    # preserve non-subject entries like time_resampled
    for k, v in seg_dict.items():
        if not isinstance(v, dict):
            out[k] = v

    return out

def filter_segments(
    seg_dict,
    muscle_keys,
    consistency_mode='any',
    rms_threshold=0.5,  
    mean_threshold=0.5,
):

    # --- compute global bands ---
    all_segs_by_muscle = {}
    for muscle in muscle_keys:
        pooled = []
        for subj, data in seg_dict.items():
            if not isinstance(data, dict):
                continue
            for seg in data.get(muscle, []):
                pooled.append(np.asarray(seg))
        if pooled:
            all_segs_by_muscle[muscle] = np.stack(pooled)

    bands = {}
    for muscle, arr in all_segs_by_muscle.items():
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        bands[muscle] = {
            'mean': mean,
            'std': std,
            'lo': mean - 2.5 * std,
            'hi': mean + 2.5 * std,
        }

    # --- filter ---
    filtered_dict = {}
    dropped = []

    for subj, data in seg_dict.items():
        if not isinstance(data, dict):
            filtered_dict[subj] = data
            continue

        n_segs = len(next(
            (v for v in data.values() if isinstance(v, list) and len(v) > 0), []
        ))

        keep_mask = np.ones(n_segs, dtype=bool)
        bad_muscles_per_seg = [[] for _ in range(n_segs)]

        for muscle in muscle_keys:
            if muscle not in data or muscle not in bands:
                continue
            lo = bands[muscle]['lo']
            hi = bands[muscle]['hi']

            for i, seg in enumerate(data[muscle]):
                seg = np.asarray(seg)
                # excess is how far outside the band each point is (0 if inside)
                excess = np.maximum(seg - hi, 0) + np.maximum(lo - seg, 0)
                rms_excess = np.sqrt(np.mean(excess**4))   # catches peaky outliers
                lo2 = bands[muscle]['mean'] - 2 * bands[muscle]['std']
                hi2 = bands[muscle]['mean'] + 2 * bands[muscle]['std']
                pct_outside = np.mean((seg < lo2) | (seg > hi2))  # fraction of timepoints beyond ±2 std
                if rms_excess > rms_threshold or pct_outside > mean_threshold:
                    bad_muscles_per_seg[i].append(muscle)

        for i in range(n_segs):
            bad = bad_muscles_per_seg[i]
            is_bad = (consistency_mode == 'any' and len(bad) > 0) or \
                     (consistency_mode == 'all' and len(bad) == len(muscle_keys))
            if is_bad:
                keep_mask[i] = False
                dropped.append((subj, i, bad))

        filtered_dict[subj] = {}
        for key, val in data.items():
            if isinstance(val, list) and len(val) == n_segs:
                filtered_dict[subj][key] = [s for s, keep in zip(val, keep_mask) if keep]
            else:
                filtered_dict[subj][key] = val

    return filtered_dict, dropped, bands


#Plotting funcions
def plot_achilles_segments(achilles_resampled, time_resampled, figsize=(10, 10), linewidth=2):
    """
    achilles_resampled: list/array of segments, each shape (T,)
    time_resampled: typically resampled_segs['time_resampled'] where time_resampled[0] is (T,)
    """
    fig = plt.figure(figsize=figsize)

    num_segments = len(achilles_resampled)
    x = time_resampled[0] * 100  # percent stance

    for i in range(num_segments):
        plt.plot(x, achilles_resampled[i], linewidth=linewidth)

    plt.ylabel("Achilles Muscle Force (N)", fontsize=18)
    plt.xlabel("Percent Normalized Stance", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_muscle_grid(
    resampled_muscles,
    time_resampled,
    muscle_keys,
    nrows=11,
    ncols=3,
    figsize=(15, 25),
    alpha=0.4,
    linewidth=2,
    plot_mean=True,
    mean_linewidth=3,
):
    """
    resampled_muscles[muscle] -> list or array of (N_segments, T)
    time_resampled[0] -> (T,) in [0,1]
    muscle_keys -> list of base muscle names (strings)
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    x = time_resampled[0] * 100  # percent stance

    for i, ax in enumerate(axes):
        if i >= len(muscle_keys):
            ax.axis("off")
            continue

        key = muscle_keys[i]
        segments = resampled_muscles[key]

        # plot all segments
        for seg in segments:
            ax.plot(x, seg, linewidth=linewidth, color="#A2C7E7", alpha=alpha)

        # mean curve
        if plot_mean and len(segments) > 0:
            Y = np.asarray(segments)
            if Y.ndim == 2:
                ax.plot(x, Y.mean(axis=0), linewidth=mean_linewidth)

        # labels
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Percent Normalized Stance", fontsize=12)
        if i % ncols == 0:
            ax.set_ylabel("Muscle Force (N)", fontsize=12)

        # auto title from key
        title = key.replace("_", " ").title()
        ax.set_title(title, fontsize=18)

        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

    plt.tight_layout()
    plt.show()

def summarize_segments(log_text):
    """
    Takes raw multiline text like 'Trial:OA1_80_2, start time:0.6065'
    and returns a summary count of segments per subject.
    """
    # Extract subject codes using regex (e.g. OA1, OA2, OA17, etc.)
    subjects = re.findall(r"Trial:(OA\d+)_", log_text)
    counts = Counter(subjects)

    # Print nicely formatted summary
    print(f"{'Subject':<8} {'# Segments':>10}")
    print("-" * 22)
    for subj, n in sorted(counts.items(), key=lambda x: int(x[0][2:])):  # sort by number
        print(f"{subj:<8} {n:>10}")
    print("-" * 22)
    print(f"{'Total':<8} {sum(counts.values()):>10}")

    return counts
