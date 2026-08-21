import numpy as np
import optuna
import pytest
import torch
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial

from api import inference
from models.architectures import build_model

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def fixture_paths(tmp_path, monkeypatch):
    paths = {
        'model_dir': tmp_path / 'models',
        'splits_dir': tmp_path / 'splits',
        'test_data_dir': tmp_path / 'test_data',
    }
    for p in paths.values():
        p.mkdir()
    monkeypatch.setattr(inference, '_paths', lambda: paths)
    return paths


def _write_lstm_fixture(paths, study_name='Test_v1.0_lstm', n_inputs=5, n_outputs=3):
    """A real (tiny) LSTM model + matching .pth + matching Optuna study, so
    load_model/run_on_test_data exercise the real build_model/load_state_dict/
    optuna.load_study path end to end rather than mocking it away."""
    bp = {'hidden_size': 4, 'num_layers': 1, 'dropout_rate': 0.0}
    model = build_model('lstm', bp, n_inputs, n_outputs, torch.device('cpu'))
    torch.save(model.state_dict(), paths['model_dir'] / f'{study_name}.pth')

    major = study_name.split('_v')[1].split('.')[0]
    db_path = paths['model_dir'] / f'optuna_v{major}.db'
    study = optuna.create_study(study_name=study_name, storage=f'sqlite:///{db_path}')
    study.add_trial(create_trial(
        params=bp,
        distributions={k: CategoricalDistribution([v]) for k, v in bp.items()},
        value=0.0,
    ))
    return bp


def _write_npz_fixture(paths, filename='Test_v1_test_data.npz', n_samples=4, seq_len=10,
                       n_inputs=5, n_outputs=3):
    rng = np.random.default_rng(0)
    X = rng.random((n_samples, seq_len, n_inputs)).astype(np.float32)
    y = rng.random((n_samples, seq_len, n_outputs)).astype(np.float32)
    output_keys = np.array([f'out{i}' for i in range(n_outputs)])
    np.savez(paths['splits_dir'] / filename, X_test=X, y_test=y, output_keys=output_keys)
    return X, y, output_keys


# ── Filename parsing ─────────────────────────────────────────────────────────

@pytest.mark.parametrize('study_name,expected_model,expected_major', [
    ('Silder_mixed_v1.1_lstm', 'lstm', '1'),
    ('Silder_mixed_v2.0_cnn_lstm', 'cnn_lstm', '2'),
    ('Uhlrich_v3.0_baseline_lstm_attn', 'lstm_attn', '3'),
    ('Uhlrich_v3.0_baseline_transformer', 'transformer', '3'),
])
def test_infer_model_name_and_major_version(study_name, expected_model, expected_major):
    """Model name and major version both parse correctly regardless of the
    optional stratum suffix (e.g. '_baseline') sitting between them."""
    assert inference._infer_model_name(study_name) == expected_model
    assert inference._infer_major_version(study_name) == expected_major


def test_infer_model_name_unknown_suffix_raises():
    with pytest.raises(ValueError):
        inference._infer_model_name('Some_v1.0_unknown_arch')


def test_infer_major_version_missing_pattern_raises():
    with pytest.raises(ValueError):
        inference._infer_major_version('no_version_here_lstm')


# ── File listing ──────────────────────────────────────────────────────────────

def test_list_pth_files_empty_when_dir_missing(tmp_path, monkeypatch):
    missing = tmp_path / 'does_not_exist'
    monkeypatch.setattr(inference, '_paths', lambda: {
        'model_dir': missing, 'splits_dir': missing, 'test_data_dir': missing,
    })
    assert inference.list_pth_files() == []
    assert inference.list_test_data_files() == []


def test_list_pth_files_finds_fixture(fixture_paths):
    _write_lstm_fixture(fixture_paths)
    assert inference.list_pth_files() == ['Test_v1.0_lstm.pth']


def test_list_test_data_files_covers_both_dirs(fixture_paths):
    _write_npz_fixture(fixture_paths, 'a_test_data.npz')
    (fixture_paths['test_data_dir'] / 'b_test_only.npz').touch()
    (fixture_paths['test_data_dir'] / 'irrelevant.txt').touch()
    assert inference.list_test_data_files() == ['a_test_data.npz', 'b_test_only.npz']


# ── peek_output_keys ─────────────────────────────────────────────────────────

def test_peek_output_keys_does_not_require_a_model(fixture_paths):
    """Populating the output-channel dropdown shouldn't need a loaded model."""
    _, _, output_keys = _write_npz_fixture(fixture_paths)
    assert inference.peek_output_keys('Test_v1_test_data.npz') == list(output_keys)


def test_peek_output_keys_missing_file_raises(fixture_paths):
    with pytest.raises(FileNotFoundError):
        inference.peek_output_keys('nope.npz')


# ── load_model / run_on_test_data ───────────────────────────────────────────

def test_load_model_end_to_end(fixture_paths):
    _write_lstm_fixture(fixture_paths)
    model, device = inference.load_model('Test_v1.0_lstm.pth', n_inputs=5, n_outputs=3)
    assert device.type == 'cpu'
    with torch.no_grad():
        out = model(torch.rand(1, 10, 5))
    assert tuple(out.shape) == (1, 10, 3)


def test_load_model_missing_weight_file_raises(fixture_paths):
    with pytest.raises(FileNotFoundError):
        inference.load_model('nope.pth', n_inputs=5, n_outputs=3)


def test_run_on_test_data_end_to_end(fixture_paths):
    _write_lstm_fixture(fixture_paths)
    X, y, output_keys = _write_npz_fixture(fixture_paths)
    result = inference.run_on_test_data('Test_v1.0_lstm.pth', 'Test_v1_test_data.npz')
    assert result['output_keys'] == list(output_keys)
    assert result['n_samples'] == X.shape[0]
    assert result['y_pred'].shape == y.shape
    assert result['y_true'].shape == y.shape


def test_run_on_test_data_subject_ids_absent_is_none(fixture_paths):
    """The npz fixture here (like most non-Uhlrich datasets) doesn't carry
    subject_ids -- must degrade to None, not KeyError."""
    _write_lstm_fixture(fixture_paths)
    _write_npz_fixture(fixture_paths)
    result = inference.run_on_test_data('Test_v1.0_lstm.pth', 'Test_v1_test_data.npz')
    assert result['subject_ids'] is None


def test_run_on_test_data_subject_ids_passed_through_when_present(fixture_paths):
    _write_lstm_fixture(fixture_paths)
    X, y, output_keys = _write_npz_fixture(fixture_paths)
    np.savez(fixture_paths['splits_dir'] / 'Test_v1_test_data.npz', X_test=X, y_test=y,
             output_keys=output_keys, subject_ids=np.array(['S1', 'S1', 'S2', 'S2']))
    result = inference.run_on_test_data('Test_v1.0_lstm.pth', 'Test_v1_test_data.npz')
    assert result['subject_ids'] == ['S1', 'S1', 'S2', 'S2']


def test_run_on_test_data_missing_npz_raises(fixture_paths):
    _write_lstm_fixture(fixture_paths)
    with pytest.raises(FileNotFoundError):
        inference.run_on_test_data('Test_v1.0_lstm.pth', 'nope.npz')


def test_run_on_test_data_shape_mismatch_raises_not_silently_wrong(fixture_paths):
    """A model trained for a different n_outputs than the test data provides
    must fail loudly (state_dict shape mismatch), not silently truncate/pad."""
    _write_lstm_fixture(fixture_paths, n_outputs=3)
    _write_npz_fixture(fixture_paths, n_outputs=7)
    with pytest.raises(RuntimeError):
        inference.run_on_test_data('Test_v1.0_lstm.pth', 'Test_v1_test_data.npz')


# ── summarize_run ────────────────────────────────────────────────────────────

def test_summarize_run_matches_metrics_doc_shape(fixture_paths):
    """Same field names/shape as one model's entry in a saved metrics.yaml
    (Compare_Models.ipynb's export) -- that's what lets api/render.py's
    table/per-output-chart builders work on a live run unmodified."""
    _write_lstm_fixture(fixture_paths)
    output_keys = _write_npz_fixture(fixture_paths)[2]
    run = inference.run_on_test_data('Test_v1.0_lstm.pth', 'Test_v1_test_data.npz')
    summary = inference.summarize_run(run)

    for key in ('rrmse_w', 'r2_mean', 'dice_mean', 'auc_mean', 'mae_overall'):
        assert isinstance(summary[key], float)
    for field in ('per_output_rrmse', 'per_output_mae', 'per_output_r2',
                  'per_output_dice', 'per_output_auc'):
        assert set(summary[field].keys()) == set(output_keys)
        assert all(isinstance(v, float) for v in summary[field].values())


def test_summarize_run_perfect_prediction_gives_ideal_values():
    """A sanity check independent of any model/file fixture: feeding
    y_pred == y_true through summarize_run's math should read as 'perfect'."""
    y = np.random.default_rng(0).random((5, 10, 3)).astype(np.float32) + 0.1  # avoid all-zero
    run = {'y_true': y, 'y_pred': y.copy(), 'output_keys': ['a', 'b', 'c']}
    summary = inference.summarize_run(run)
    assert summary['rrmse_w'] == pytest.approx(0.0, abs=1e-5)
    assert summary['r2_mean'] == pytest.approx(1.0, abs=1e-5)
    assert summary['dice_mean'] == pytest.approx(1.0, abs=1e-5)
    assert all(v == pytest.approx(0.0, abs=1e-5) for v in summary['per_output_rrmse'].values())


# ── resolve_subset / is_in_reported_subset ──────────────────────────────────
# Both read config.yaml's reporting: section directly (like data_access.py's
# versions/ dir tests) -- static, checked-in content, safe to test against
# the real file rather than needing a fixture.

def test_resolve_subset_v1_uses_its_own_override():
    """v1 predates hip JRFs -- its subset must not include them."""
    subset = inference.resolve_subset('1')
    assert 'hip_fy' not in subset
    assert 'knee_fy' in subset


def test_resolve_subset_falls_back_to_primary_outputs_for_unlisted_version():
    """No V99_outputs exists -- must fall back rather than KeyError."""
    subset = inference.resolve_subset('99')
    assert subset == inference.resolve_subset('99')   # stable/deterministic
    assert 'hip_fy' in subset   # matches primary_outputs, unlike V1


def test_resolve_subset_excludes_addmagmid_from_every_version():
    """addmagMid is a real predicted output (signals.outputs) but isn't part
    of any reported subset -- the concrete example used in the glossary."""
    assert 'addmagMid' not in inference.resolve_subset('1')
    assert 'addmagMid' not in inference.resolve_subset('2')


def test_is_in_reported_subset_true_for_subset_member():
    # Pure filename parsing + the real config.yaml -- no model/npz needed.
    assert inference.is_in_reported_subset('Test_v1.0_lstm.pth', 'knee_fy') is True


def test_is_in_reported_subset_false_for_non_subset_output():
    assert inference.is_in_reported_subset('Test_v1.0_lstm.pth', 'addmagMid') is False


def test_is_in_reported_subset_false_rather_than_raising_for_unparseable_filename():
    assert inference.is_in_reported_subset('not_a_versioned_filename.pth', 'knee_fy') is False


# ── compute_gastroc_soleus_predictions ──────────────────────────────────────

def _write_ratio_npz_fixture(paths, filename, subject_ids, gasmed_val, soleus_val,
                             n_inputs=5, seq_len=6):
    n_samples = len(subject_ids)
    rng = np.random.default_rng(0)
    X = rng.random((n_samples, seq_len, n_inputs)).astype(np.float32)
    y = np.zeros((n_samples, seq_len, 3), dtype=np.float32)
    y[:, :, 0] = gasmed_val
    y[:, :, 1] = soleus_val
    output_keys = np.array(['gasmed', 'soleus', 'other'])
    np.savez(paths['test_data_dir'] / filename, X_test=X, y_test=y, output_keys=output_keys,
             subject_ids=np.array(subject_ids))


def test_compute_gastroc_soleus_predictions_end_to_end(fixture_paths):
    _write_lstm_fixture(fixture_paths, study_name='Test_v1.0_baseline_lstm', n_outputs=3)
    _write_lstm_fixture(fixture_paths, study_name='Test_v1.0_retention_lstm', n_outputs=3)
    subjects = ['S1', 'S1', 'S2', 'S2']
    _write_ratio_npz_fixture(fixture_paths, 'Test_v1_baseline_test_only.npz', subjects,
                             gasmed_val=0.4, soleus_val=0.4)
    _write_ratio_npz_fixture(fixture_paths, 'Test_v1_retention_test_only.npz', subjects,
                             gasmed_val=0.2, soleus_val=0.6)

    out = inference.compute_gastroc_soleus_predictions('Test_v1.0', 'test', ['lstm'])
    assert out['subjects'] == ['S1', 'S2']
    for subj in out['subjects']:
        gt = out['ground_truth'][subj]
        assert gt['baseline'] == pytest.approx(0.5)   # 0.4 / (0.4 + 0.4)
        assert gt['retention'] == pytest.approx(0.25)  # 0.2 / (0.2 + 0.6)
        for regime in ('base_trained', 'ret_trained'):
            point = out['predictions']['lstm'][regime][subj]
            # Untrained/random weights -- just check it's a real computed
            # ratio, not that it lands in any particular range.
            assert isinstance(point['baseline'], float)
            assert isinstance(point['retention'], float)


def test_compute_gastroc_soleus_predictions_omits_model_without_saved_weights(fixture_paths):
    """No pooled-trained .pth exists (see PLAN A follow-up) -- must be
    omitted from predictions, not raise."""
    _write_lstm_fixture(fixture_paths, study_name='Test_v1.0_baseline_lstm', n_outputs=3)
    subjects = ['S1', 'S2']
    _write_ratio_npz_fixture(fixture_paths, 'Test_v1_baseline_test_only.npz', subjects, 0.4, 0.4)
    _write_ratio_npz_fixture(fixture_paths, 'Test_v1_retention_test_only.npz', subjects, 0.2, 0.6)

    out = inference.compute_gastroc_soleus_predictions('Test_v1.0', 'test', ['lstm'])
    assert 'ret_trained' not in out['predictions']['lstm']   # no retention-trained .pth saved
    assert 'base_trained' in out['predictions']['lstm']


def test_compute_gastroc_soleus_predictions_missing_npz_returns_empty(fixture_paths):
    out = inference.compute_gastroc_soleus_predictions('Test_v1.0', 'test', ['lstm'])
    assert out == {'subjects': [], 'ground_truth': {}, 'predictions': {}}


def test_compute_gastroc_soleus_predictions_unparseable_version_returns_empty(fixture_paths):
    out = inference.compute_gastroc_soleus_predictions('no-version-here', 'test', ['lstm'])
    assert out == {'subjects': [], 'ground_truth': {}, 'predictions': {}}
