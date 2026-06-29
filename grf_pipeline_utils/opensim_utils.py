import gc
import os
import re

import numpy as np
import opensim as osim
import pandas as pd

from .signal_processing import filter_id, filter_ik


def scale_generic(root_dir: str, mass: float, static_pose_filename: str, scaling_dir: str):
    dir = root_dir
    os.chdir(dir)
    subject_id = os.path.basename(static_pose_filename).split('_walk_static')[0]
    #scale generic model
    setup = 'generic_scale_setup.xml'
    scale_tool = osim.ScaleTool(setup)
    scale_tool.setSubjectMass(mass)
    scale_tool.setName(f'{subject_id}_scaled')
    #set the path to the generic model
    model_maker = scale_tool.getGenericModelMaker()
    model_maker.setModelFileName('Models/RajagopalModified_generic.osim')
    #model_maker.setMarkerSetFileName()
    #set marker file for model scaler
    model_scaler = scale_tool.getModelScaler()
    model_scaler.setMarkerFileName(static_pose_filename)
    #access marker placer object and set inputs
    marker_placer = scale_tool.getMarkerPlacer()
    marker_placer.setStaticPoseFileName(static_pose_filename)
    marker_placer.setOutputModelFileName(os.path.join(scaling_dir, f'{subject_id}_scaled.osim'))
    scale_tool.run()
    del scale_tool
    gc.collect()

def inverse_kinmatics(root_dir: str, tracking_data_filepath: str, model: osim.Model, 
                      ik_raw_dir: str, ik_filtered_dir: str, setup_dir: str):
    os.chdir(root_dir)
    subj_trial_speed = os.path.basename(tracking_data_filepath).replace('_transformed.trc', '')
    
    ik_tool = osim.InverseKinematicsTool(os.path.join(setup_dir, 'generic_ik_setup.xml'))
    ik_tool.set_report_marker_locations(False)
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(tracking_data_filepath)
    ik_tool.setOutputMotionFileName(os.path.join(ik_raw_dir, f'{subj_trial_speed}_ik.mot'))
    ik_tool.run()
    
    filter_ik(
        os.path.join(ik_raw_dir, f'{subj_trial_speed}_ik.mot'),
        os.path.join(ik_filtered_dir, f'{subj_trial_speed}_ik_filtered.mot')
    )
    del ik_tool
    gc.collect()

def inverse_dynamics(root_dir: str, force_data_filepath: str, tracking_data_filepath: str, 
                     model: osim.Model, loads_dir: str, id_raw_dir: str, 
                     id_filtered_dir: str, setup_dir: str):
    os.chdir(root_dir)
    subj_trial_speed = os.path.basename(tracking_data_filepath).replace('_transformed.trc', '')
    
    loads = osim.ExternalLoads(os.path.join(setup_dir, 'generic_externalLoads.xml'), True)
    loads.setDataFileName(force_data_filepath)
    loads_path = os.path.join(loads_dir, f'{subj_trial_speed}_externalLoads.xml')
    loads.printToXML(loads_path)
    
    id_tool = osim.InverseDynamicsTool(os.path.join(setup_dir, 'generic_id_setup.xml'))
    id_tool.setModel(model)
    id_tool.setExternalLoadsFileName(loads_path)
    ik_file = os.path.join(id_raw_dir.replace('ID/raw', 'IK/filtered'), 
                           f'{subj_trial_speed}_ik_filtered.mot')
    id_tool.setCoordinatesFileName(ik_file)
    id_tool.set_results_directory(id_raw_dir)
    id_tool.setOutputGenForceFileName(f'{subj_trial_speed}_id.mot')
    id_tool.run()
    
    filter_id(
        os.path.join(id_raw_dir, f'{subj_trial_speed}_id.mot'),
        os.path.join(id_filtered_dir, f'{subj_trial_speed}_id_filtered.mot')
    )
    del id_tool
    gc.collect()

def split_scaling_subject_logs(raw_text):
    """
    Split a combined OpenSim scaling output into per-subject text blocks.
    Returns a dict: {subject_name: [list of lines]}
    """
    subject_blocks = {}
    current_subject = None
    current_lines = []

    for line in raw_text.splitlines():
        match = re.search(r'\[info\]\s*Processing subject\s+(\w+)_scaled', line)
        if match:
            # Save the previous subject’s lines if we were collecting one
            if current_subject and current_lines:
                subject_blocks[current_subject] = current_lines
            # Start a new subject
            current_subject = match.group(1)
            current_lines = []
        elif current_subject:
            current_lines.append(line)

    # Save last one
    if current_subject and current_lines:
        subject_blocks[current_subject] = current_lines

    return subject_blocks


def parse_scaling_block(lines):
    """
    Parse a single subject's scaling output block into structured info.
    """
    info = {
        "marker_file": None,
        "scaling_factors": {},
        "missing_markers": [],
        "unused_measurements": [],
        "marker_error_rms": None,
        "marker_error_max": None,
        "marker_error_max_marker": None
    }

    current_measurement = None
    for line in lines:
        # Marker file
        if "Loaded marker file" in line:
            match = re.search(r'(/Users[^\s]+\.trc)', line)
            if match:
                info["marker_file"] = match.group(1)

        # Measurement name
        if "[info] Measurement" in line:
            current_measurement = re.search(r"'(.+)'", line).group(1)

        # Scale factor
        elif "overall scale factor" in line:
            factor = float(re.search(r'=\s*([0-9.]+)', line).group(1))
            if current_measurement:
                info["scaling_factors"][current_measurement] = factor

        # Missing marker warnings
        elif "Marker" in line and "not found" in line:
            marker = re.search(r"Marker ([^ ]+) ", line)
            if marker:
                info["missing_markers"].append(marker.group(1))

        # Unused measurements
        elif "measurement not used" in line:
            meas = re.search(r"'([^']+)'", line)
            if meas:
                info["unused_measurements"].append(meas.group(1))

        # Marker error summary
        elif "marker error: RMS" in line:
            rms = re.search(r'RMS = ([0-9.]+)', line)
            max_ = re.search(r'max = ([0-9.]+)', line)
            matches = re.findall(r'\(([^)]+)\)', line)
            marker = matches[-1] if matches else None
            if rms and max_:
                info["marker_error_rms"] = float(rms.group(1))
                info["marker_error_max"] = float(max_.group(1))
                info["marker_error_max_marker"] = marker if marker else None

    return info


def parse_combined_scaling_output(file_path):
    """
    Main function to parse a full multi-subject OpenSim scaling output.
    Returns dict of subject_name → parsed info.
    """
    try:
        with open(file_path, 'r') as file:
            text = file.read()
            print('Read scaling log')
    except FileNotFoundError:
        print(f'Error: the file {file_path} does not exist')
    except Exception as e:
        print(f'An error has occured: {e}')

    subjects = split_scaling_subject_logs(text)
    all_results = {}

    for subj, lines in subjects.items():
        all_results[subj] = parse_scaling_block(lines)

    return all_results


def parse_ik_block(block):
    """Extract metrics from one IK tool run."""
    data = {
        "subject": None,
        "model": None,
        "frames": 0,
        "mean_rms": None,
        "std_rms": None,
        "max_rms": None,
        "mean_max": None,
        "std_max": None,
        "max_marker_error": None,
        "max_marker_name": None,
        "completion_time_s": None,
    }

    # Try to find the last model name before Running tool
    model_match = re.search(r"MODEL:\s*(\S+)", block)
    if model_match:
        data["model"] = model_match.group(1)
        subj_match = re.match(r"(OA\d+)", data["model"])
        data["subject"] = subj_match.group(1) if subj_match else None

    # Extract all frame errors
    frame_lines = re.findall(
        r"\[info\]\s*Frame\s+\d+.*?RMS\s*=\s*([0-9.]+).*?max\s*=\s*([0-9.]+)\s*\(([^)]+)\)",
        block
    )
    if frame_lines:
        rms_vals = np.array([float(f[0]) for f in frame_lines])
        max_vals = np.array([float(f[1]) for f in frame_lines])

        data["frames"] = len(frame_lines)
        data["mean_rms"] = np.mean(rms_vals)
        data["std_rms"] = np.std(rms_vals)
        data["max_rms"] = np.max(rms_vals)
        data["mean_max"] = np.mean(max_vals)
        data["std_max"] = np.std(max_vals)
        data["max_marker_error"] = np.max(max_vals)

    # Completion info
    completion = re.search(
        r"\[info\]\s*InverseKinematicsTool completed\s+(\d+)\s+frames\s+in\s+([0-9.]+)\s+second",
        block
    )
    if completion:
        data["frames"] = int(completion.group(1))
        data["completion_time_s"] = float(completion.group(2))

    return data


def parse_full_ik_log(file_path, trial_names):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
            print('Read inverse kinematics log')
    except FileNotFoundError:
        print(f'Error: the file {file_path} does not exist')
    except Exception as e:
        print(f'An error has occured: {e}')
    pattern = (
        r"(MODEL:\s*\S+[\s\S]*?InverseKinematicsTool completed\s+\d+\s+frames\s+in\s+[0-9.]+\s+second\(s\)\.)" # noqa: E501
    )
    matches = re.findall(pattern, text, flags=re.S)
    parsed = [parse_ik_block(m) for m in matches]

    # count trials per subject
    subj_counter = {}
    trial_ids = []
    for r in parsed:
        subj = r.get("subject", "unknown")
        subj_counter[subj] = subj_counter.get(subj, 0) + 1
        trial_ids.append(f"{subj}_trial{subj_counter[subj]}")
    n_matches = len(parsed)
    n_trials = len(matches)
    n = min(n_matches, n_trials)
    df = pd.DataFrame(parsed)
    df.insert(0, "trial_name", trial_names[:n])
    return df
