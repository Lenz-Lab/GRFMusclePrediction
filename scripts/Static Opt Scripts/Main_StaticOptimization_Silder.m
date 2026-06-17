% Custom static optimization code. Author: Scott Uhlrich, Stanford
% University, 2020. Please cite:
% Uhlrich, S.D., Jackson, R.W., Seth, A., Kolesar, J.A., Delp S.L.
% Muscle coordination retraining inspired by musculoskeletal simulations
% reduces knee contact force. Sci Rep 12, 9842 (2022).
% https://doi.org/10.1038/s41598-022-13386-9
%
% Adapted for OA/Y walking dataset by Brian Keller.
% Merged from Main_StaticOptimization.m (reference) and
% Silder_StaticOptimization.m. Key fixes applied:
%   - JointReaction now runs on ALL joints (jointNames = 'all')
%   - setStartTime/setEndTime use actual trial times (not hardcoded 1s)
%   - lb/ub arrays correctly sized to nMuscles+nFreeCoords
%   - Removed stray forceReport.end comment inconsistency

%%
function [] = Main_StaticOptimization_Silder()

clear all; close all; format compact; clc; fclose all;

% =========================================================================
% USER CONFIGURATION
% =========================================================================

% Base directory containing all subject folders and results
data_dir = '/Users/briankeller/Desktop/GRFMusclePrediction/';

% Add source utilities
addpath(genpath('src'));

% Subjects to process
subject_dir = {'OA1',  'OA2',  'OA4',  'OA5',  'OA7',  'OA8',  'OA9',  'OA10', ...
               'OA11', 'OA12', 'OA13', 'OA14', 'OA15', 'OA17', 'OA18', 'OA19', 'OA20', ...
               'OA22', 'OA23', 'OA24', 'OA25', ...
               'Y1',  'Y2',  'Y4',  'Y5',  'Y6',  'Y7',  'Y8',  'Y9',  ...
               'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17', ...
               'Y18', 'Y19', 'Y20', 'Y21', 'Y22'};

% Trials to process per subject
trial_names = {'_80_1',  '_80_2',  '_80_3',  '_80_4',  '_80_5',  ...
               '_100_1', '_100_2', '_100_3', '_100_4', '_100_5', ...
               '_120_1', '_120_2', '_120_3', '_120_4', '_120_5'};

% Geometry folder for OpenSim models. Leave as [] to use OpenSim default.
geometryPath = [];

% =========================================================================
% STATIC OPTIMIZATION PARAMETERS  (same for all subjects/trials)
% =========================================================================

% Lowpass filter frequency for IK coordinates (Hz). Set to -1 for no filter.
INPUTS.filtFreq = 6;

% Flags
INPUTS.appendActuators           = true;   % Append reserve actuators at all coordinates
INPUTS.appendForces              = true;   % Append GRFs as external forces
INPUTS.deleteContralateralMuscles = false; % Replace contralateral muscles with strong reserves (faster SO)
INPUTS.useEmgRatios              = false;  % Constrain muscle ratios to EMG
INPUTS.useEqualMuscles           = false;  % Constrain muscle pairs to equal activation
INPUTS.useEmgConstraints         = false;  % Constrain activations to track EMG directly
INPUTS.changePassiveForce        = false;  % Modify passive force-length curves
INPUTS.ignoreTendonCompliance    = false;  % Make all tendons rigid

% Leg to actuate with muscles when deleteContralateralMuscles = true.
% Does not matter when deleteContralateralMuscles = false.
INPUTS.leg = 'r';

% DOFs to exclude from moment-matching constraint (patella coupler etc.)
INPUTS.fixedDOFs = {'knee_angle_r_beta', 'knee_angle_l_beta'};

% EMG options (unused unless flags above are true)
INPUTS.emgFilePath           = '';
INPUTS.emgRatioPairs         = {};
INPUTS.equalMuscles          = {};
INPUTS.emgConstrainedMuscles = {};
INPUTS.emgSumThreshold       = 0;

% Cost function weights: sum(w * x^2)
INPUTS.muscleWeights              = 1;
INPUTS.reserveActuatorWeights     = 1;
INPUTS.ipsilateralActuatorStrength  = 1;
INPUTS.contralateralActuatorStrength = 100;

% Per-actuator weight overrides. weightsToOverride is a cell of partial
% names (e.g. 'gastroc' matches gastroc_r and gastroc_l).
% overrideWeights is a numeric vector the same length as weightsToOverride.
INPUTS.weightsToOverride = {};
INPUTS.overrideWeights   = [];

% Coordinates whose reserve actuator absorbs any residual ID moment
% (the optimizer does not try to balance these DOFs with muscles).
INPUTS.prescribedActuationCoords = {};

% Passive force-length curve strains [strain@zeroForce, strain@oneNormForce]
% Only used when changePassiveForce = true.
INPUTS.passiveForceStrains = [3, 4];

% =========================================================================
% EXTERNAL FORCE DEFINITIONS  (GRF application)
% =========================================================================

INPUTS.externalForceName     = {'GRF_r',         'GRF_l'};
INPUTS.applied_to_body       = {'calcn_r',        'calcn_l'};
INPUTS.force_expressed_in_body = {'ground',       'ground'};
INPUTS.force_identifier      = {'ground_force_v', '1_ground_force_v'};
INPUTS.point_expressed_in_body = {'ground',       'ground'};
INPUTS.point_identifier      = {'ground_force_p', '1_ground_force_p'};

% =========================================================================
% JOINT REACTION ANALYSIS SETTINGS
% =========================================================================
% 'all'   -> compute JRA for every joint in the model (recommended)
% or a cell array of specific joint names e.g. {'ankle_r','knee_r',...}
%
% inFrame/onBody: 'child' reports forces in each joint's child body frame.
% Change to 'ground' if you need all joints in a common frame for direct
% comparison of shear (fx/fz) components across joints.

INPUTS.jRxn.jointNames = 'all';
INPUTS.jRxn.inFrame    = 'child';
INPUTS.jRxn.onBody     = 'child';

% =========================================================================
% END OF USER CONFIGURATION
% =========================================================================

if ~isempty(INPUTS.overrideWeights)
    disp('WARNING: You are overriding some actuator weights.');
end

if ~isempty(geometryPath)
    import org.opensim.modeling.*
    org.opensim.modeling.ModelVisualizer.addDirToGeometrySearchPaths(geometryPath);
end

% -------------------------------------------------------------------------
% BATCH LOOP
% -------------------------------------------------------------------------
for i = 1:length(subject_dir)
    subject_id = subject_dir{i};

    for j = 1:length(trial_names)
        close all;
        trial      = trial_names{j};
        trial_name = [subject_id, trial];

        fprintf('\n========================================\n');
        fprintf('Processing: %s\n', trial_name);
        fprintf('========================================\n');

        % --- File paths ---------------------------------------------------
        INPUTS.trialname    = trial_name;
        INPUTS.forceFilePath  = [data_dir, 'data/processed/Silder/', trial_name, '_transformed.mot'];
        INPUTS.ikFilePath     = [data_dir, 'results/Silder/IK/filtered/', trial_name, '_ik_filtered.mot'];
        INPUTS.idFilePath     = [data_dir, 'results/Silder/ID/filtered/', trial_name, '_id_filtered.mot'];
        INPUTS.outputFilePath = [data_dir, 'results/Silder/SO/', trial_name, '/'];
        INPUTS.modelDir       = [data_dir, 'results/Silder/Scaling/'];
        INPUTS.modelName     = [subject_id, '_scaled.osim'];

        % --- Auto-detect trial time range from force file -----------------
        import org.opensim.modeling.*
        try
            temp_storage = org.opensim.modeling.Storage(INPUTS.forceFilePath);
            time_col = ArrayDouble();
            temp_storage.getTimeColumn(time_col);
            time_vec = str2num(time_col); %#ok<ST2NM>
            INPUTS.startTime = time_vec(1);
            INPUTS.endTime   = time_vec(end);
        catch ME
            warning('Could not read force file for %s: %s\nSkipping trial.', trial_name, ME.message);
            continue
        end

        % --- Run static optimization --------------------------------------
        try
            StaticOptimizationAPIVectorized(INPUTS);
        catch ME
            warning('Static optimization failed for %s:\n%s\nSkipping trial.', trial_name, ME.message);
            continue
        end

        % --- Save a copy of this settings script to the output folder -----
        try
            mkdir(INPUTS.outputFilePath);
            src  = strcat(mfilename('fullpath'), '.m');
            dest = [INPUTS.outputFilePath, 'staticOpt_settings.m'];
            copyfile(src, dest);
        catch
            % Non-fatal: just warn if the copy fails
            warning('Could not save settings file to %s', INPUTS.outputFilePath);
        end

    end % trial loop
end % subject loop

end % Main_StaticOptimization_OAY
