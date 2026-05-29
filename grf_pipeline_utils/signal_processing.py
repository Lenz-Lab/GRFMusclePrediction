from re import L
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def lowpass_filter_df(df: pd.DataFrame, cutoff: float, fs: float, order: int = 2):
    """
    Fourth order low-pass butterworth filter 

    Parameters
    ----------
    df : data frame to be filtered
    cutoff : cutoff frequency for low-pass filtering
    fs : data sampling frequency
    order : filter order, doubled effectively due to use of filtfilt


    Returns
    -------
    filters i/p dataframe in-place
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    #create filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # type: ignore
    #apply to dataframe
    cols = df.columns
    for col in cols:
        if col in df.columns:
            df[col] = filtfilt(b, a, df[col].astype(float))
    return df

def filter_ik(input_path: str, output_path: str): 
    """
    Filters inverse kinematic data

    Parameters
    ----------
    input_path : file path to raw inverse kinematics results
    output_path : destination path for filtered IK 

    Returns
    -------
    doesn't return anything, just writes a new file of filtered IK
    """
    with open(input_path, 'r') as f:
        header = [next(f) for _ in range(10)]
        ik_df = pd.read_csv(f, sep = '\t')
    #separate time col
    time_col = ik_df['time']
    ik_df = ik_df.drop('time', axis = 1)
    #filter @ 6hz with a 100 hz sampling rate
    lowpass_filter_df(ik_df, 6, 100)
    #write data to ouptut file
    ik_df.insert(0, 'time', time_col)
    with open(output_path, 'w') as f:
        for line in header:
            f.write(line)
        ik_df.to_csv(f, sep='\t', index=False, lineterminator='\n')

def filter_id(input_path: str, output_path: str): 
    """
    Filters inverse dynamics data

    Parameters
    ----------
    input_path : file path to raw inverse dynamics results
    output_path : destination path for filtered ID 

    Returns
    -------
    doesn't return anything, just writes a new file of filtered ID
    """
    with open(input_path, 'r') as f:
        header = [next(f) for _ in range(6)]
        id_df = pd.read_csv(f, sep = '\t')
    #separate time col
    time_col = id_df['time']
    id_df = id_df.drop('time', axis = 1)
    #filter @ 6hz with a 100 hz sampling rate
    lowpass_filter_df(id_df, 6, 100)
    #write data to ouptut file
    id_df.insert(0, 'time', time_col)
    with open(output_path, 'w') as f:
        for line in header:
            f.write(line)
        id_df.to_csv(f, sep='\t', index=False, lineterminator='\n')

def detect_foot_and_stance(tracking_df: pd.DataFrame, force_df: pd.DataFrame, threshold: float, trial_name: str, pickle_path: str):
    """
    Function for re-naming ground reaction force data. Compares center of pressure data during gait cycles to heel marker x coordinates and assigns ground reaction forces to left or right foot

    Parameters
    ----------
    tracking_df : dataframe containing rotated and scaled tracking data of x coordinates for heel markers
    force_df : dataframe containing rotated and scaled ground reaction force data, point data, and ground torque data
    threshold : force threshold for extracting gait cycles
    trial_name : string containing the name of the subject, speed, and trial number. Used as a key in the dictionary that stores stance phase times for each trial
    pickle_path : string specifying the file directory in which prcoessed grf pickles are stored for later use

    Returns
    -------
    final_df : re-formatted dataframe of ground reaction force data that is either right foot or left foot, rather than one of three force plates
    stance_segs : dictionary with keys as trial name and values as left and right stance segment times
    """
    #initialize dictionary for current trial
    stance_segs = {trial_name: {'left': [], 'right': []}}
    #helper function for finding start/end indices from a binary mask
    def mask_to_segs(mask):
        starts = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
        ends = np.where(np.diff(mask.astype(int)) == -1)[0] + 1
        #edge cases
        if ends.size and starts.size and ends[0] < starts[0]:
            ends = ends[1:]
        if starts.size > ends.size:
            starts = starts[:len(ends)]
        return list(zip(starts, ends))
    
    #initialize new data structures for grf data (left and right vs 1, 2, and 3)
    left_force = np.zeros((len(force_df), 3))
    right_force = np.zeros((len(force_df), 3))
    left_cop = np.zeros((len(force_df), 4))
    right_cop = np.zeros((len(force_df), 4))
    left_torque = np.zeros((len(force_df), 3))
    right_torque = np.zeros((len(force_df), 3))
    time_force = np.asarray(force_df['time'].values)
    tracking_time = np.asarray(tracking_df['Time'].values)
    l_heel = np.asarray(tracking_df['L.Heel'].values)
    r_heel = np.asarray(tracking_df['R.Heel'].values)
    prev_foot = 'unknown'
    #generate dataframes from raw force data
    for i in range(1, 4):
        fx = np.asarray(force_df[f'FX{i}'].values)
        fy = np.asarray(force_df[f'FY{i}'].values)
        fz = np.asarray(force_df[f'FZ{i}'].values)
        copx = np.asarray(force_df[f'X{i}'].values)
        copy = np.asarray(force_df[f'Y{i}'].values)
        copz = np.asarray(force_df[f'Z{i}'].values)
        tx = np.asarray(force_df[f'MX{i}'].values)
        ty = np.asarray(force_df[f'MY{i}'].values)
        tz = np.asarray(force_df[f'MZ{i}'].values)
        
        on_ground = fy > threshold
        trial_segs = mask_to_segs(on_ground)
        if not trial_segs:
            continue
        #trim segments that are not stances
        min_duration = 0.5
        valid_segs = []
        for (s, e) in trial_segs:
            duration = time_force[e] - time_force[s]
            if duration >= min_duration:
                valid_segs.append((s,e))
        trial_segs = valid_segs
        #loop through valid segments to reassign grf columns
        for (s, e) in trial_segs:
            #get times from tracking and force data that correspond with valid segment start/ends
            force_t1, force_t2 = time_force[s], time_force[e]
            force_tseg = time_force[s:e]               # length N_force
            # pick the tracking samples over the whole segment window
            tracking_mask = (tracking_time >= force_t1) & (tracking_time <= force_t2)
            if tracking_mask.sum() < 2:
                continue

            t_tr = tracking_time[tracking_mask]
            l_tr = l_heel[tracking_mask]
            r_tr = r_heel[tracking_mask]

            # interpolate heel x to force timestamps
            lheel_force = np.interp(force_tseg, t_tr, l_tr)
            rheel_force = np.interp(force_tseg, t_tr, r_tr)

            copx_seg = copx[s:e]

            # distances between cop and heel markers for side classification 
            dist_L = np.mean(np.abs(copx_seg - lheel_force))
            dist_R = np.mean(np.abs(copx_seg - rheel_force))

            #get vertical reaction force segment
            fy_seg   = fy[s:e]

            copx_norm = np.zeros_like(copx_seg)
            # find peaks with adaptive prominence + spacing
            #min_prom_frac = 0.15
            min_dist_s = 0.2
            #width_s = 0.1
            fs = 2000
            #prom = min_prom_frac * np.max(fy_seg) if np.max(fy_seg) > 0 else 0
            dist = int(min_dist_s * fs)
            #width = width_s * fs

            peaks, props = find_peaks(fy_seg, height = 0.5 * np.max(fy_seg), distance=dist)
            N = len(fy_seg)

            if peaks.size >= 2:
                pk1 = peaks[0]
                pk2 = peaks[-1]
            elif peaks.size == 1:
                print(f'one peak found!, trial: {trial_name}, start time: {s}')
                pk1 = peaks[0]
                pk2 = N
            else:
                print(f'no peaks found! trial: {trial_name}, start time: {s}')
                pk1 = int(np.argmax(fy_seg))
                pk2 = N
            copx_norm[pk1:pk2] = copx_seg[pk1:pk2] - copx_seg[pk1]

            if dist_L + 1e-6 < dist_R:
                side = 'left'
            elif dist_R + 1e-6 < dist_L:
                side = 'right'
            elif prev_foot == 'right':
                side = 'left'
            else:
                side = 'right'
            #update grf data and segment dictionary according to which foot is assigned to the stance segment
            if side == 'left':
                prev_foot = 'left'
                left_force[s:e, 0] += fx[s:e]
                left_force[s:e, 1] += fy[s:e]
                left_force[s:e, 2] += fz[s:e]
                left_cop[s:e, 0] += copx[s:e]
                left_cop[s:e, 1] += copy[s:e]
                left_cop[s:e, 2] += copz[s:e]
                left_cop[s:e, 3] += copx_norm
                left_torque[s:e, 0] += tx[s:e]
                left_torque[s:e, 1] += ty[s:e]
                left_torque[s:e, 2] += tz[s:e]
                stance_segs[trial_name]['left'].append((force_t1, force_t2))
            else:
                prev_foot = 'right'
                right_force[s:e, 0] += fx[s:e]
                right_force[s:e, 1] += fy[s:e]
                right_force[s:e, 2] += fz[s:e]
                right_cop[s:e, 0] += copx[s:e]
                right_cop[s:e, 1] += copy[s:e]
                right_cop[s:e, 2] += copz[s:e]
                right_cop[s:e, 3] += copx_norm
                right_torque[s:e, 0] += tx[s:e]
                right_torque[s:e, 1] += ty[s:e]
                right_torque[s:e, 2] += tz[s:e]
                stance_segs[trial_name]['right'].append((force_t1, force_t2))

    # Combine into a final output DataFrame
    final_df = pd.DataFrame({
        'time': time_force,
        # Right forces
        'ground_force_vx': right_force[:, 0],
        'ground_force_vy': right_force[:, 1],
        'ground_force_vz': right_force[:, 2],
         # Right COP
        'ground_force_px': right_cop[:, 0],
        'ground_force_py': right_cop[:, 1],
        'ground_force_pz': right_cop[:, 2],
        'ground_force_new_px' : right_cop[:, 3],
        # Left forces
        '1_ground_force_vx': left_force[:, 0],
        '1_ground_force_vy': left_force[:, 1],
        '1_ground_force_vz': left_force[:, 2],
        # Left COP
        '1_ground_force_px': left_cop[:, 0],
        '1_ground_force_py': left_cop[:, 1],
        '1_ground_force_pz': left_cop[:, 2],
        '1_ground_force_new_px' : left_cop[:, 3],
        # Right torques
        'ground_torque_x': right_torque[:, 0],
        'ground_torque_y': right_torque[:, 1],
        'ground_torque_z': right_torque[:, 2],
        # Left torques
        '1_ground_torque_x': left_torque[:, 0],
        '1_ground_torque_y': left_torque[:, 1],
        '1_ground_torque_z': left_torque[:, 2],
       
    })
    #write our final df to a .pkl file for faster loading because we will need it later
    if pickle_path == '':
        pickle_path = '/Users/briankeller/Desktop/GRFMuscleModel/Old_Young_Walking_Data/transformed/grf_pickles/' + trial_name
    final_df.to_pickle(pickle_path)
    final_df = final_df.drop(columns=['1_ground_force_new_px', 'ground_force_new_px'])
    return final_df, stance_segs        

def process_hjc_trc(input_path: str, output_path: str, markers_to_drop: list):
    """
    Transforms raw tracking data with a rotation and unit conversion (mm -> m)

    Parameters
    ----------
    input_path : file path to raw tracking data file
    output_path : file path for writing transformed tracking data 

    Returns
    -------
    dataframe containing x coordinates of left and right heel markers
    """
    #Load file
    with open(input_path, 'r') as f:
        header = [next(f) for _ in range(5)]
        metadata_values = header[2].strip().split()
        if 'mm' in metadata_values:
            units_index = metadata_values.index('mm')
            metadata_values[units_index] = 'm'
            header[2] = '\t'.join(metadata_values) + '\n'
        marker_names = header[3].strip().split()
        col_names = header[4].strip().split()
        tracking_df = pd.read_csv(f, sep='\t', header = None)  
    header[3] = '\t\t' + '\t'.join(marker_names) + '\n'
    #drop unnamed columns
    tracking_df.dropna(axis=1, how='all', inplace=True)
    #Split metadata (Frame # and Time)
    meta_cols = tracking_df.columns[:2]  
    meta_df = tracking_df[meta_cols]
    meta_df = meta_df.rename(columns={meta_df.columns[0]: '', meta_df.columns[1]: ''})
    tracking_df = tracking_df.drop(columns=meta_cols)
    tracking_df.columns = col_names
    #get index corresponding to heel markers
    r_idx = marker_names.index('R.Heel')
    l_idx = marker_names.index('L.Heel')
    #get corresponding x position columns for heel markers
    r_x_col = (r_idx - 2) * 3
    l_x_col = (l_idx - 2) * 3 
    #extract position data for heel markers
    r_heel = tracking_df.iloc[:, r_x_col] / 1000
    l_heel = tracking_df.iloc[:, l_x_col] / 1000
    heel_df = pd.concat([meta_df.iloc[:, 1], r_heel, l_heel], axis=1)
    heel_df.columns = ['Time','R.Heel', 'L.Heel']
    #associate marker data with names
    marker_names = marker_names[2 :]
    marker_dict = {}
    for i, name in enumerate(marker_names):
        start_col = i * 3
        marker_dict[name] = tracking_df.iloc[:, start_col:start_col + 3] 
    #drop specified markers
    cols_to_drop = []
    for name in markers_to_drop:
        if name in marker_dict:
            sub_df = marker_dict[name]
            cols_to_drop += list(sub_df.columns)
    tracking_df = tracking_df.drop(columns=cols_to_drop)
    filtered_marker_names = [m for m in marker_names if m not in markers_to_drop]
    final_marker_names = ['Frame#', 'Time'] + filtered_marker_names
    header[3] = '\t'.join(final_marker_names) + '\n'  
    # Reassign headers after dropping columns
    new_headers = []
    num_markers_remaining = tracking_df.shape[1] // 3
    for i in range(1, num_markers_remaining + 1):
        new_headers.extend([f'X{i}', f'Y{i}', f'Z{i}'])
    tracking_df.columns = new_headers
    #adjust header to reflect new number of markers
    parts = header[2].strip().split()
    parts[3] = str(num_markers_remaining)  # Update NumMarkers
    header[2] = '\t'.join(parts) + '\n'
    #Define Rotation
    angle = np.radians(-90)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])
    marker_ids = sorted(set(int(col[1:]) for col in tracking_df.columns if col.startswith('X')))
    for idx in marker_ids:
        try:
            coords = tracking_df[[f'X{idx}', f'Y{idx}', f'Z{idx}']].astype(float).values
            rotated = coords @ R.T
            tracking_df[f'X{idx}'], tracking_df[f'Y{idx}'], tracking_df[f'Z{idx}'] = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        except KeyError:
            continue
    #convert marker positions to meters
    tracking_df[tracking_df.columns] = tracking_df[tracking_df.columns].astype(float) / 1000
    tracking_df = tracking_df.round(5)
    #recombine metadata
    df_final = pd.concat([meta_df, tracking_df], axis=1)
    #write final df to new file
    with open(output_path, 'w') as f:
        for i, line in enumerate(header):
            if i == 4:
                continue
            f.write(line)
        df_final.to_csv(f, sep='\t', index=False, lineterminator='\n')
    #return dataframe of heel marker x coords
    return heel_df

def process_grf(input_path: str):
    """
    Transforms raw ground reaction force by adding X and Y ground torques and applying a rotation to force, torque and center of pressure data. Also filters and converts center of pressure data from mm to m. 

    Parameters
    ----------
    input_path : file path to raw ground reaction force data file

    Returns
    -------
    dataframe of transformed force data
    """
    #load data from file into a header and a dataframe
    with open(input_path, 'r') as f:
        header = [next(f) for _ in range(4)]  
        forces_df = pd.read_csv(f, sep='\t')  
    #compute time column from samples (sampling rate = 2000 hz)
    samples = forces_df.iloc[:, 0].astype(float)
    forces_df.insert(0, 'time', samples / 2000.0)
    #drop samples column
    forces_df.drop(columns=forces_df.columns[1], inplace=True)
    #add 0's for x and y torques
    for i in range(1, 4):
        forces_df[f'MX{i}'] = 0.0
        forces_df[f'MY{i}'] = 0.0
        if f'MZ{i}' not in forces_df.columns:
            forces_df[f'MZ{i}'] = 0.0
    #define rotation
    angle = np.radians(-90)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])
    #apply rotation and scaling (point data mm to m)
    for i in range(1, 4):
        F = forces_df[[f'FX{i}', f'FY{i}', f'FZ{i}']].astype(float).values
        forces_df[[f'FX{i}', f'FY{i}', f'FZ{i}']] = F @ R.T
        P = forces_df[[f'X{i}', f'Y{i}', f'Z{i}']].astype(float).values
        P = (P @ R.T) / 1000.0
        forces_df[[f'X{i}', f'Y{i}', f'Z{i}']] = P
        T = forces_df[[f'MX{i}', f'MY{i}', f'MZ{i}']].astype(float).values
        forces_df[[f'MX{i}', f'MY{i}', f'MZ{i}']] = (T @ R.T) / 1000
    #separate COP and torque data
    cop_cols = [col for col in forces_df.columns if col[:1] in ['X', 'Y', 'Z'] and col[1:].isdigit()]
    cop_df = forces_df[cop_cols].copy()
    torque_cols = [col for col in forces_df.columns if col[:2] in['MX', 'MY', 'MZ'] and col[2:].isdigit()]
    torque_df=forces_df[torque_cols].copy()
    #filter COP and torque data @ 6hz with a 2000 hz sampling rate
    lowpass_filter_df(cop_df, 6, 2000)
    lowpass_filter_df(torque_df, 6, 2000)
    forces_df = forces_df.drop(cop_cols, axis = 1)
    forces_df = forces_df.drop(torque_cols, axis=1)
    #add COP's back to full dataframe
    final_df = pd.concat([cop_df, forces_df, torque_df], axis = 1)
    #ensure time col starts from 0 and is furthest to the left
    #final_df['time'] = final_df['time'] - final_df['time'].iloc[0]
    final_cols = ['time'] + [c for c in final_df.columns if c != 'time']
    final_df = final_df[final_cols]
    return final_df

def preprocess_trc_grf(trc_ip: str, trc_op: str, markers_to_drop: list,  grf_ip: str, grf_op: str, grf_pickle_path: str):
    """
    Function that calls file preprocessing and center of pressure detection functions. Writes transformed and re-formatted ground reation force data to a .mot file

    Parameters
    ----------
    trc_ip : file path to raw tracking data file
    trc_op : file path for transformed tracking data to be written 
    grf_ip : file path to raw ground reaction force data file
    grf_op : file path for transforemd force data to be written

    Returns
    -------
    Dictionary containing start and end times of stance segments \n
    Also writes a new .mot file for the transformed force data after it is reformatted into L/R rather than plates 1-3
    """
    #parse output name for trial name
    after_trans = trc_op.split('/transformed/')[1]
    trial_name = after_trans.split('_transformed')[0]
    #load data into dataframes
    heel_df = process_hjc_trc(trc_ip, trc_op, markers_to_drop)
    grf_df = process_grf(grf_ip)
    #call stance detection function
    final_grf_df, stance_segs = detect_foot_and_stance(heel_df, grf_df, 1, trial_name, grf_pickle_path)
    #write finalized grf data to .mot file with proper header
    mot_header_lines = [
        f"{grf_op.split('/')[-1]}",  
        "version=1",
        f"nRows={len(final_grf_df)}",
        f"nColumns={final_grf_df.shape[1]}",
        "inDegrees=yes",
        "endheader\n"
    ]
    #write data to .mot file
    with open(grf_op, 'w') as f:
        for line in mot_header_lines:
            f.write(line + '\n')
        final_grf_df.to_csv(f, sep='\t', index=False, lineterminator='\n')
    return stance_segs
