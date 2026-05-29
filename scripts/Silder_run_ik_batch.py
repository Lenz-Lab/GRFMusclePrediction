import os
from grf_pipeline_utils.signal_processing import *
from grf_pipeline_utils.opensim_utils import *


root_dir = '/Users/briankeller/Desktop/GRFMuscleModel/Old_Young_Walking_Data/'
OA_subjects = [f'OA{i}' for i in range(1, 26)]
Y_subjects = [f'Y{i}' for i in range(1, 23)]
subjects = OA_subjects + Y_subjects
bad_subjects = ['OA3', 'OA6', 'OA15', 'OA16', 'OA21', 'Y3']
trial_names = []
for bs in bad_subjects:
    subjects.remove(bs)
speeds = ['80', '100', '120']
n_trials = 5
subject_trials = {}
for subj in subjects:
    subj_dir = os.path.join(root_dir, subj, 'Walking/Files_W_HJCs/')
    output_dir = os.path.join(root_dir, 'transformed/')
    # Initialize nested dict
    if subj[0] == 'O':
        subject_trials[subj] = {
            'static': {
                'input': os.path.join(subj_dir, f'{subj}_walk_static1.trc'),
                'output': os.path.join(output_dir, f'{subj}_walk_static1_transformed.trc')
            },
            'tracking': [],
            'forces': [],
            
        }
    else:
        subject_trials[subj] = {
            'static': {
                'input': os.path.join(subj_dir, f'{subj}_walking_static1.trc'),
                'output': os.path.join(output_dir, f'{subj}_walk_static1_transformed.trc')
            },
            'tracking': [],
            'forces': [],
            
        }
    # Populate tracking and force trial lists
    for spd in speeds:
        for i in range(1, n_trials + 1):
            trial_name = f'{subj}_{spd}_{i}'
            trial_names.append(str(trial_name))
            tracking_in = os.path.join(subj_dir, f'{trial_name}.trc')
            tracking_out = os.path.join(output_dir, f'{trial_name}_transformed.trc')
            force_in = os.path.join(subj_dir, f'{trial_name}.forces')
            force_out = os.path.join(output_dir, f'{trial_name}_transformed.mot')

            subject_trials[subj]['tracking'].append({'input': tracking_in, 'output': tracking_out})
            subject_trials[subj]['forces'].append({'input': force_in, 'output': force_out})
            
#Call IK on all trials
for subj, data in subject_trials.items():
    model = osim.Model(root_dir + f'Results/Scaling/{subj}_scaled.osim')
    for trc in data['tracking']:
        inverse_kinmatics(root_dir= root_dir, tracking_data_filepath=trc['output'], model=model)