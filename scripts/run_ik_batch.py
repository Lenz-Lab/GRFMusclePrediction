import os
from grf_pipeline_utils.signal_processing import *
from grf_pipeline_utils.opensim_utils import *
import yaml

repo_root = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(repo_root)

with open(os.path.join(repo_root, 'config.yaml')) as f:
    cfg = yaml.safe_load(f)

root_dir        = os.path.join(repo_root, cfg['silder']['data_root'])
output_dir = os.path.join(repo_root, cfg['paths']['processed_dir'], 'Silder')
scaling_dir     = os.path.join(repo_root, cfg['silder']['results']['scaling'])
ik_raw_dir      = os.path.join(repo_root, cfg['silder']['results']['ik_raw'])
ik_filtered_dir = os.path.join(repo_root, cfg['silder']['results']['ik_filtered'])
setup_dir       = os.path.join(repo_root, cfg['silder']['opensim_setup_dir'])

OA_subjects = [f"OA{i}" for i in cfg['silder']['OA_subjects']]
Y_subjects  = [f"Y{i}" for i in cfg['silder']['Y_subjects']]
subjects    = OA_subjects + Y_subjects
speeds      = cfg['silder']['speeds']
n_trials    = cfg['silder']['n_trials']

trial_names    = []
subject_trials = {}
for subj in subjects:
    subj_dir = os.path.join(root_dir, subj, 'Walking/Files_W_HJCs/')
    # OA subjects use '_walk_static1' while Y subjects use '_walking_static1'
    static_name = f'{subj}_walk_static1.trc' if subj[0] == 'O' else f'{subj}_walking_static1.trc'
    subject_trials[subj] = {
        'static': {
            'input':  os.path.join(subj_dir, static_name),
            'output': os.path.join(output_dir, f'{subj}_walk_static1_transformed.trc')
        },
        'tracking': [],
        'forces':   []
    }
    for spd in speeds:
        for i in range(1, n_trials + 1):
            trial_name = f'{subj}_{spd}_{i}'
            trial_names.append(trial_name)
            subject_trials[subj]['tracking'].append({
                'input':  os.path.join(subj_dir, f'{trial_name}.trc'),
                'output': os.path.join(output_dir, f'{trial_name}_transformed.trc')
            })
            subject_trials[subj]['forces'].append({
                'input':  os.path.join(subj_dir, f'{trial_name}.forces'),
                'output': os.path.join(output_dir, f'{trial_name}_transformed.mot')
            })

# Call IK on all trials
for subj, data in subject_trials.items():
    model = osim.Model(os.path.join(scaling_dir, f'{subj}_scaled.osim'))
    for trc in data['tracking']:
        inverse_kinmatics(
            root_dir=root_dir,
            tracking_data_filepath=trc['output'],
            model=model,
            ik_raw_dir=ik_raw_dir,
            ik_filtered_dir=ik_filtered_dir,
            setup_dir=setup_dir
        )