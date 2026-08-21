"""Loads trained models and runs live inference against existing test data.

Mirrors notebooks/Compare_Models.ipynb's loading cell exactly (same Optuna
study-name convention, same .pth loading call) rather than inventing a new
one, so a model that loads in that notebook loads here.
"""
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import torch
import yaml

from grf_pipeline_utils.eval_utils import (
    calc_auc_per_output,
    calc_dice_per_output,
    calc_gastroc_soleus_ratio,
    calc_mae_overall,
    calc_mae_per_output,
    calc_r2_per_output,
    calc_rrmse_per_output,
    calc_rrmse_weighted,
)
from models.architectures import build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / 'config.yaml'

_MODEL_NAMES = ('lstm_attn', 'cnn_lstm', 'transformer', 'lstm')   # longest suffixes first


def _paths() -> dict[str, Path]:
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    p = cfg['paths']
    return {
        'model_dir': REPO_ROOT / p['model_dir'],
        'splits_dir': REPO_ROOT / p['splits_dir'],
        'test_data_dir': REPO_ROOT / p['test_data_dir'],
    }


def resolve_subset(major: str) -> list[str]:
    """Same version-resolution the notebooks use for SUBSET_KEYS: reporting.
    V{major}_outputs if that version has its own override (e.g. v1 excludes
    hip JRFs), else reporting.primary_outputs. Used to gate the AUC-overlap
    shading in /predict to outputs that are actually part of the standard
    reported subset, not just whatever's in the .npz."""
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    reporting = cfg['reporting']
    return reporting.get(f'V{major}_outputs', reporting['primary_outputs'])


def is_in_reported_subset(pth_filename: str, output_key: str) -> bool:
    """Whether `output_key` is part of the version-resolved reported subset
    for this .pth's dataset version. Returns False (gate just stays off)
    rather than raising if the filename doesn't parse -- this only controls
    an optional visual, not correctness."""
    try:
        major = _infer_major_version(pth_filename.removesuffix('.pth'))
    except ValueError:
        return False
    return output_key in resolve_subset(major)


def list_pth_files() -> list[str]:
    model_dir = _paths()['model_dir']
    if not model_dir.is_dir():
        return []
    return sorted(p.name for p in model_dir.glob('*.pth'))


def list_test_data_files() -> list[str]:
    paths = _paths()
    files = []
    if paths['splits_dir'].is_dir():
        files += [p.name for p in paths['splits_dir'].glob('*_test_data.npz')]
    if paths['test_data_dir'].is_dir():
        files += [p.name for p in paths['test_data_dir'].glob('*_test_only.npz')]
    return sorted(set(files))


def _find_file(candidates: list[str], *tokens: str) -> Optional[str]:
    """First filename containing every token, case-insensitive -- locates a
    .pth/.npz by naming convention (e.g. version + condition + architecture)
    without assuming exact casing (results/metrics' 'uhlrich' vs. models/'s
    'Uhlrich', for one)."""
    for f in candidates:
        fl = f.lower()
        if all(tok.lower() in fl for tok in tokens):
            return f
    return None


def _resolve_test_data_path(npz_filename: str) -> Optional[Path]:
    paths = _paths()
    for d in (paths['splits_dir'], paths['test_data_dir']):
        candidate = d / npz_filename
        if candidate.is_file():
            return candidate
    return None


def _infer_model_name(study_name: str) -> str:
    for name in _MODEL_NAMES:
        if study_name.endswith(f'_{name}'):
            return name
    raise ValueError(
        f'Could not infer model architecture from {study_name!r} -- '
        f'expected it to end with one of {_MODEL_NAMES}',
    )


def _infer_major_version(study_name: str) -> str:
    m = re.search(r'_v(\d+)\.\d+', study_name)
    if not m:
        raise ValueError(
            f'Could not find a "_v<major>.<minor>" version in {study_name!r}',
        )
    return m.group(1)


def peek_output_keys(npz_filename: str) -> list[str]:
    """Just the output_keys array -- cheap, doesn't touch X_test/y_test, so the
    UI can populate an output-channel dropdown before running any inference."""
    path = _resolve_test_data_path(npz_filename)
    if path is None:
        raise FileNotFoundError(f'No test data file named {npz_filename!r}')
    with np.load(path, allow_pickle=True) as data:
        return [str(k) for k in data['output_keys']]


def load_model(pth_filename: str, n_inputs: int, n_outputs: int):
    """Returns (model, device). Reproduces Compare_Models.ipynb's loading cell:
    Optuna study_name == the .pth filename's stem, best_params -> build_model,
    then load_state_dict from the .pth."""
    paths = _paths()
    weight_path = paths['model_dir'] / pth_filename
    if not weight_path.is_file():
        raise FileNotFoundError(f'No weight file named {pth_filename!r}')

    study_name = pth_filename.removesuffix('.pth')
    model_name = _infer_model_name(study_name)
    major = _infer_major_version(study_name)
    optuna_db_path = paths['model_dir'] / f'optuna_v{major}.db'
    optuna_db = f'sqlite:///{optuna_db_path}'

    study = optuna.load_study(study_name=study_name, storage=optuna_db)
    bp = study.best_params

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(model_name, bp, n_inputs, n_outputs, device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def run_on_test_data(pth_filename: str, npz_filename: str) -> dict[str, Any]:
    """Runs the model against every sample in the given test .npz. Returns
    output_keys, y_true, y_pred (all samples -- callers slice into one
    sample/output at a time for display rather than re-running inference)."""
    path = _resolve_test_data_path(npz_filename)
    if path is None:
        raise FileNotFoundError(f'No test data file named {npz_filename!r}')

    data = np.load(path, allow_pickle=True)
    X_test, y_test = data['X_test'], data['y_test']
    output_keys = [str(k) for k in data['output_keys']]
    subject_ids = [str(s) for s in data['subject_ids']] if 'subject_ids' in data else None

    model, device = load_model(pth_filename, X_test.shape[2], len(output_keys))
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_t).cpu().numpy()

    return {
        'output_keys': output_keys,
        'y_true': y_test,
        'y_pred': y_pred,
        'n_samples': X_test.shape[0],
        'subject_ids': subject_ids,
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Test-set-wide metrics for one run() result, shaped exactly like one
    model's entry in a saved metrics_doc['models'] (see the "Save versioned
    metrics" cell in Compare_Models.ipynb) -- so a live run can reuse the same
    comparison-table/per-output-chart rendering (api/render.py) as a saved
    results/metrics/*.yaml, instead of a separate bespoke display."""
    y_true, y_pred, output_keys = run['y_true'], run['y_pred'], run['output_keys']
    rrmse_per = calc_rrmse_per_output(y_true, y_pred, verbose=False)
    mae_per = calc_mae_per_output(y_true, y_pred, verbose=False)
    r2_per = calc_r2_per_output(y_true, y_pred, verbose=False)
    dice_per = calc_dice_per_output(y_true, y_pred, verbose=False)
    auc_per = calc_auc_per_output(y_true, y_pred, verbose=False)
    return {
        'rrmse_w': float(calc_rrmse_weighted(y_true, y_pred)),
        'r2_mean': float(r2_per.mean()),
        'dice_mean': float(dice_per.mean()),
        'auc_mean': float(auc_per.mean()),
        'mae_overall': float(calc_mae_overall(y_true, y_pred)),
        'per_output_rrmse': dict(zip(output_keys, (float(v) for v in rrmse_per))),
        'per_output_mae': dict(zip(output_keys, (float(v) for v in mae_per))),
        'per_output_r2': dict(zip(output_keys, (float(v) for v in r2_per))),
        'per_output_dice': dict(zip(output_keys, (float(v) for v in dice_per))),
        'per_output_auc': dict(zip(output_keys, (float(v) for v in auc_per))),
    }


def _group_mean_by_subject(values: np.ndarray, subject_ids: list[str]) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for subj, v in zip(subject_ids, values):
        sums[subj].append(float(v))
    return {subj: sum(vs) / len(vs) for subj, vs in sums.items()}


def compute_gastroc_soleus_predictions(
    version: str, dataset: str, model_names: list[str],
) -> dict[str, Any]:
    """Live per-subject gastroc:soleus force_ratio -- ground truth, plus
    base-trained/ret-trained predicted -- for the cross-stratum transfer
    experiment. CrossVal.ipynb's ratio_recovery block already summarizes
    this as aggregate stats (bias, spread_ratio, ...); this surfaces the
    same underlying quantity per subject instead, via the real model and
    real test data, so a reader can see which individual subjects drive
    those aggregate numbers. Reuses calc_gastroc_soleus_ratio (eval_utils.py)
    -- the same function the notebook uses -- so these can't silently drift
    from the saved numbers.

    Runs real inference; larger architectures can take up to roughly a
    minute on CPU, which is why api/main.py gates this behind an explicit
    '?live=1' opt-in rather than running it on every page view.

    pooled-trained is never included: no pooled .pth is saved to disk yet,
    so it's left out rather than guessed at -- not a bug, just missing
    weights.

    Returns {'subjects': [...], 'ground_truth': {subject: {baseline, retention}},
    'predictions': {model_name: {'base_trained': {subject: {...}}, 'ret_trained': {...}}}}.
    An architecture/regime this can't find a matching .pth for is simply
    omitted, not an error."""
    try:
        major = _infer_major_version(version)
    except ValueError:
        return {'subjects': [], 'ground_truth': {}, 'predictions': {}}

    npz_files = list_test_data_files()
    # '_test_only.npz' (not '..._test_data.npz') specifically -- the file that
    # holds every cycle for these subjects, which is what the notebook's
    # ratio_recovery n=10 stats are computed over; the '_test_data.npz' split
    # only has some subjects/cycles and would silently disagree with them.
    baseline_npz = _find_file(npz_files, dataset, f'v{major}', 'baseline', 'test_only')
    retention_npz = _find_file(npz_files, dataset, f'v{major}', 'retention', 'test_only')
    if baseline_npz is None or retention_npz is None:
        return {'subjects': [], 'ground_truth': {}, 'predictions': {}}

    ground_truth: dict[str, dict[str, float]] = defaultdict(dict)
    subjects: set[str] = set()
    for condition, npz_filename in (('baseline', baseline_npz), ('retention', retention_npz)):
        path = _resolve_test_data_path(npz_filename)
        with np.load(path, allow_pickle=True) as data:
            if 'subject_ids' not in data:
                continue
            subject_ids = [str(s) for s in data['subject_ids']]
            output_keys = [str(k) for k in data['output_keys']]
            true_ratio = calc_gastroc_soleus_ratio(data['y_test'], output_keys)
        for subj, val in _group_mean_by_subject(true_ratio, subject_ids).items():
            ground_truth[subj][condition] = val
            subjects.add(subj)

    pth_files = list_pth_files()
    predictions: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for model_name in model_names:
        for regime, trained_on in (('base_trained', 'baseline'), ('ret_trained', 'retention')):
            pth = _find_file(pth_files, version, trained_on, model_name)
            if pth is None:
                continue   # e.g. pooled-trained -- no weights saved to disk
            per_subject: dict[str, dict[str, float]] = {}
            conditions = (('baseline', baseline_npz), ('retention', retention_npz))
            for condition, npz_filename in conditions:
                run = run_on_test_data(pth, npz_filename)
                if run['subject_ids'] is None:
                    continue
                pred_ratio = calc_gastroc_soleus_ratio(run['y_pred'], run['output_keys'])
                for subj, val in _group_mean_by_subject(pred_ratio, run['subject_ids']).items():
                    per_subject.setdefault(subj, {})[condition] = val
            if per_subject:
                predictions.setdefault(model_name, {})[regime] = per_subject

    return {
        'subjects': sorted(subjects),
        'ground_truth': dict(ground_truth),
        'predictions': predictions,
    }
