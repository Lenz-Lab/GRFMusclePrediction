# GRF Muscle Prediction
[![CI](https://github.com/Lenz-Lab/GRFMusclePrediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Lenz-Lab/GRFMusclePrediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

Predicting lower-limb muscle forces and joint reaction forces from ground reaction forces (GRF) during walking using sequence and attention based deep learning models.

---

## Problem Statement

Measuring muscle forces in vivo is invasive and impractical in clinical settings. Ground reaction forces, by contrast, are routinely measured with force plates and are increasingly accessible via instrumented insoles and walkways. This project trains sequence models to predict 37 lower-limb muscle forces and 6 joint reaction forces directly from GRF and center-of-pressure (COP) data collected during walking — enabling non-invasive estimation of internal musculoskeletal loading from externally measurable quantities.

The pipeline runs from raw motion capture and force plate data through OpenSim musculoskeletal simulation (scaling, inverse kinematics, inverse dynamics, static optimization) to produce labeled training data, then trains and evaluates four sequence model architectures: LSTM, LSTM with multi-head attention, CNN-LSTM, and a Transformer encoder. Models are evaluated both within-dataset and cross-dataset to assess generalizability across populations and experimental conditions.

---

## Pipeline Overview

flowchart TD
    A[Raw marker + force plate data] --> B[OpenSim preprocessing\nScaling · IK · ID · Static Opt]
    B --> C[Gait cycle segmentation\nsignal_processing.py]
    C --> D[Normalization + filtering\ndata_utils.py]
    D --> E[Train / val / test split\nSplit_Single_Dataset.ipynb]
    E --> F[Hyperparameter tuning\nTune_Models.ipynb · Optuna]
    F --> G[Final model training\nLSTM · LSTM+Attn · CNN-LSTM · Transformer]
    G --> H[Evaluation + cross-dataset testing\nCompare_Models.ipynb]
```

---

## Datasets

### Silder et al. (2008)

**Source:** Silder A, Heiderscheit B, Thelen DG. Active and passive contributions to joint kinetics during walking in older adults. *Journal of Biomechanics*, 41(7):1520–1527, 2008. https://doi.org/10.1016/j.jbiomech.2008.02.016

This dataset contains motion capture and discrete force plate data from older adult (OA) and young adult (YA) walkers across three speeds (80%, 100%, and 120% of preferred walking speed), with five trials per speed per subject. Raw data consists of `.trc` marker files and `.forces` ground reaction force files.

**Preprocessing pipeline** (see `notebooks/Silder_Preprocessing.ipynb` and `scripts/run_ik_batch.py`):

All OpenSim steps were run using the Rajagopal et al. full-body model. The preprocessing pipeline consists of:

1. **Coordinate transformation** — marker and force data are transformed to the reference frame used by OpenSim and written to `results/Silder/transformed/`.  Force data is also remapped from a 3-channel format (force plates 1, 2, and 3) to a 2-channel format (left and right foot) as required by OpenSim's external loads and static optimization tools
2. **Model scaling** — subject-specific models are generated from static trial marker data using OpenSim's Scale Tool
3. **Inverse kinematics** — joint angles are computed from marker trajectories and filtered at 6 Hz
4. **Inverse dynamics** — joint moments are computed from filtered kinematics and GRF data
5. **Static optimization** — muscle forces are estimated by minimizing the sum of squared muscle activations subject to moment-balancing constraints, using the custom vectorized solver from Uhlrich et al. (2022)

**Gait cycle segmentation** (see `notebooks/Silder_Batch_Processing.ipynb`):

Stance phases are detected from vertical GRF using a threshold-based foot contact algorithm. Each stance phase is treated as one segment. Segments are resampled to 100 timepoints via linear interpolation, normalized by subject body mass, and filtered to remove outliers beyond 2.5 standard deviations of the population mean for any muscle.

**Final data structure:**

Processed segments are stored as subject-keyed pickle dictionaries with the following signal keys:

```
Inputs  (6):  grf_x, grf_y, grf_z, cop_x, cop_y, cop_z
Outputs (44): [37 muscle forces] + knee_fx, knee_fy, knee_fz, ankle_fx, ankle_fy, ankle_fz
```

---

### Ulrich et al. (2022)

**Source:** Uhlrich SD, Jackson RW, Seth A, Kolesar JA, Delp SL. Muscle coordination retraining inspired by musculoskeletal simulations reduces knee contact force. *Scientific Reports*, 12(1):9842, 2022. https://doi.org/10.1038/s41598-022-13386-9

This dataset contains 10 young adult subjects performing treadmill walking across multiple feedback and retention conditions. Unlike the Silder dataset, OpenSim results (scaling, IK, ID, static optimization) are provided precomputed per subject and per trial, organized as:

```
data/Ulrich_Data/SubjectX/
├── expmtldata/grf/          ← raw force plate data
├── expmtldata/markerdata/   ← raw marker files
├── models/                  ← generic and scaled .osim files
├── Scaling/                 ← scaling setup XML
├── IK/<trial>/output/       ← IK results
├── ID/<trial>/output/       ← ID results
└── Sopt/<trial>/output/     ← static optimization results
```

**Gait cycle segmentation** (see `notebooks/Ulrich_Batch_Processing.ipynb`):

The same segmentation, resampling, normalization, and filtering steps as the Silder pipeline are applied. Segments are stored in the same subject-keyed pickle format with identical signal keys, making the processed Ulrich data a drop-in replacement for cross-dataset evaluation.

---

### Adding a New Dataset

The pipeline is designed to be dataset-agnostic downstream of preprocessing. To incorporate a new dataset, the preprocessing script must produce a subject-keyed dictionary with the following contract:

```python
{
    'SubjectID': {
        # Inputs — normalized by body mass (N/kg), resampled to 100 timepoints
        'grf_x': [np.array(shape=(100,)), ...],   # mediolateral GRF
        'grf_y': [np.array(shape=(100,)), ...],   # vertical GRF
        'grf_z': [np.array(shape=(100,)), ...],   # anteroposterior GRF
        'cop_x': [np.array(shape=(100,)), ...],   # center of pressure
        'cop_y': [np.array(shape=(100,)), ...],
        'cop_z': [np.array(shape=(100,)), ...],

        # Outputs — muscle forces normalized by body mass (N/kg)
        'addbrev': [...], 'addlong': [...], ...   # see config.yaml signals.outputs
    }
}
```

The full list of required output keys is defined in `config.yaml` under `signals.outputs`. Output key names must match exactly — they are used to align model inputs and outputs at evaluation time and are saved alongside test datasets as `output_keys` arrays in `.npz` files. During cross-dataset evaluation in `Compare_Models`, the notebook verifies that the test dataset's `output_keys` match the model's expected outputs before running inference, preventing silent shape mismatches.

Add the new dataset's configuration (subject list, masses, trial names, directory structure) to `config.yaml` following the existing `silder` or `ulrich` sections, then add a corresponding preprocessing notebook to `notebooks/`.

---

## Models

Four sequence-to-sequence architectures are implemented in `models/architectures.py`, all taking input of shape `(batch, 100, 6)` and producing output of shape `(batch, 100, N_outputs)`:

- **LSTM** — stacked LSTM layers with a linear output projection
- **LSTM + Attention** — LSTM followed by multi-head self-attention applied to the full output sequence
- **CNN-LSTM** — two-layer 1D CNN feature extractor followed by stacked LSTM layers
- **Transformer** — standard encoder architecture with sinusoidal positional encoding

Hyperparameters for all models are tuned using Optuna with a TPE sampler. Tuning studies and final model weights are stored in `models/`.

---

## Results

*Results will be updated once final training runs are complete.*

Cross-dataset evaluation is performed by training on one dataset and evaluating on a held-out test split from the other, using `TEST_ONLY` mode in the split notebooks to package an entire dataset as a test set without train/val splitting. This tests whether models learn generalizable biomechanical relationships rather than dataset-specific patterns.

---

## Installation

This project requires Python 3.11 and OpenSim 4.x. OpenSim must be installed separately via conda — it is not available on PyPI.

```bash
# 1. Create and activate the OpenSim conda environment
conda create -n opensim_scripting python=3.11
conda activate opensim_scripting
conda install -c opensim-org opensim

# 2. Clone the repo
git clone https://github.com/Lenz-Lab/GRFMusclePrediction.git
cd GRFMusclePrediction

# 3. Install the package and dependencies
pip install -e .

# 4. Install dev dependencies (for running tests)
pip install -e ".[dev]"
```

**Note:** Data files are not included in this repository. See `data/README.md` for dataset sources and expected directory structure.

---

## Usage

### Running the pipeline

Configure paths and subject lists in `config.yaml`, then run notebooks in order:

```
Silder_Preprocessing.ipynb          ← coordinate transforms, scaling, IK, ID
scripts/run_ik_batch.py         ← batch IK for all Silder subjects
scripts/Static Opt Scripts/Main_StaticOptimization_Silder.m  ← static optimization (MATLAB + OpenSim)
Silder_Batch_Processing.ipynb  ← gait segmentation, normalization, filtering
Ulrich_Batch_Processing.ipynb  ← same for Ulrich dataset
Split_Multiple_Datasets.ipynb         ← train/val/test split
Tune_Models.ipynb            ← Optuna hyperparameter search + final training
Compare_Models.ipynb         ← evaluation and cross-dataset testing
```

### Running tests

```bash
pytest tests/ -v
```

---

## Citation

If you use this code, please cite the datasets it was developed on:

**Silder dataset:**
Silder A, Heiderscheit B, Thelen DG. Active and passive contributions to joint kinetics during walking in older adults. *Journal of Biomechanics*, 41(7):1520–1527, 2008. https://doi.org/10.1016/j.jbiomech.2008.02.016

**Ulrich / static optimization:**
Uhlrich SD, Jackson RW, Seth A, Kolesar JA, Delp SL. Muscle coordination retraining inspired by musculoskeletal simulations reduces knee contact force. *Scientific Reports*, 12(1):9842, 2022. https://doi.org/10.1038/s41598-022-13386-9
