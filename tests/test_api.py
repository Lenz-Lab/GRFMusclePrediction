import re

import numpy as np
import optuna
import torch
import yaml
from fastapi.testclient import TestClient
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial

from api import data_access, inference
from api.main import app
from models.architectures import build_model

client = TestClient(app)

# ── Fixtures — real-world shapes ────────────────────────────────────────────────
# Mirrors notebooks/Compare_Models.ipynb's metrics_doc (after the eval_type addition).
SINGLE_SPLIT_DOC = {
    'version': 'Uhlrich_v3.0',
    'eval_type': 'single_split',
    'dataset': 'uhlrich',
    'dataset_version': 'Uhlrich_v3',
    'stratum': None,
    'date': '2026-08-04',
    'models': {
        'lstm': {'test_mse': 0.01, 'rrmse_w': 0.1, 'r2_mean': 0.8},
        'transformer': {'test_mse': 0.009, 'rrmse_w': 0.09, 'r2_mean': 0.82},
    },
    'pairwise_comparisons': {'lstm_vs_transformer': {'dice_p_value': 0.03}},
}

# Mirrors notebooks/CrossVal.ipynb's metrics_doc (after the eval_type addition) —
# deliberately a different 'models' payload shape (per-fold arrays, by_group, etc.)
# to prove the API doesn't assume either shape.
CROSS_VAL_DOC = {
    'version': 'Silder_mixed_v3.0',
    'eval_type': 'cross_validation',
    'dataset': 'silder_mixed',
    'dataset_version': 'Silder_mixed_v3',
    'scheme': 'stratified 5-fold',
    'stratum': 'baseline',
    'date': '2026-08-05',
    'n_folds': 5,
    'models': {
        'lstm': {'rrmse_w_mean': 0.11, 'rrmse_w_std': 0.01,
                 'by_group': {'OA': {'rrmse_w_mean': 0.12}, 'Y': {'rrmse_w_mean': 0.10}}},
    },
}

# Mirrors CrossVal.ipynb's RUN_TRANSFER export (cross_stratum_transfer eval_type) —
# models keyed by direction (base_base/base_ret/ret_ret/ret_base) plus a
# ratio_recovery block, structurally different from both docs above.
_DIRECTION = {
    'direction': 'base->ret', 'rrmse_w_mean': 0.12, 'rrmse_w_std': 0.02,
    'clinical': {'knee_fy': {'mae_mean': 3.0, 'mae_over_threshold_mean': 0.7,
                             'threshold': 4.5}},
}
TRANSFER_DOC = {
    'version': 'Uhlrich_v3.0',
    'eval_type': 'cross_stratum_transfer',
    'dataset': 'uhlrich',
    'date': '2026-08-11',
    'n_subjects_both_phases': 10,
    'subjects_both_phases': [f'Subject{i}' for i in range(1, 11)],
    'ratio_definition': 'G = stance-averaged gasmed force ALONE; ratio = G / (G + soleus).',
    'models': {
        'lstm_attn': {
            'base_base': dict(_DIRECTION), 'base_ret': dict(_DIRECTION),
            'ret_ret': dict(_DIRECTION), 'ret_base': dict(_DIRECTION),
            'ratio_recovery': {
                'base_trained': {
                    'n': 10, 'pearson_r': 0.82, 'pearson_p': 0.0035,
                    'spearman_r': 0.96, 'spearman_p': 0.00001,
                    'bias': 0.029, 'loa_lower': -0.084, 'loa_upper': 0.142,
                    'std_pred': 0.025, 'std_gt': 0.076, 'spread_ratio': 0.32,
                },
                'ret_trained': {
                    'n': 10, 'pearson_r': 0.04, 'pearson_p': 0.91,
                    'spearman_r': 0.01, 'spearman_p': 0.99,
                    'bias': 0.025, 'loa_lower': -0.134, 'loa_upper': 0.184,
                    'std_pred': 0.032, 'std_gt': 0.076, 'spread_ratio': 0.42,
                },
                'transfer_cost': {
                    'ratio_transfer_cost_bias': 0.004, 'rrmse_w_base2ret': 0.127,
                    'rrmse_w_ret2ret': 0.127, 'rrmse_w_transfer_cost': 0.0003,
                },
            },
        },
        'cnn_lstm': {
            'base_base': dict(_DIRECTION), 'base_ret': dict(_DIRECTION),
            'ret_ret': dict(_DIRECTION), 'ret_base': dict(_DIRECTION),
            # No ratio_recovery -- transfer not run yet for this arch.
        },
    },
}


def _write_yaml(path, doc):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(doc, f)


def _patch_metrics_dir(monkeypatch, path):
    monkeypatch.setattr(data_access, 'METRICS_DIR', path)


# ── Empty / missing state ───────────────────────────────────────────────────────

def test_results_empty_when_dir_missing(monkeypatch, tmp_path):
    """A results/metrics dir that doesn't exist yet returns [], not an error."""
    _patch_metrics_dir(monkeypatch, tmp_path / 'does_not_exist')
    resp = client.get('/api/results')
    assert resp.status_code == 200
    assert resp.json() == []


def test_results_empty_when_dir_has_no_yaml_files(monkeypatch, tmp_path):
    """An existing but empty results dir also returns []."""
    _patch_metrics_dir(monkeypatch, tmp_path)
    resp = client.get('/api/results')
    assert resp.status_code == 200
    assert resp.json() == []


# ── Two real-world eval_type shapes, side by side ───────────────────────────────

def test_lists_both_eval_types_with_correct_envelopes(monkeypatch, tmp_path):
    """Distinct metrics_doc shapes (single_split vs cross_validation) both load
    fine and are individually distinguishable by eval_type/dataset/stratum."""
    _write_yaml(tmp_path / 'a_metrics.yaml', SINGLE_SPLIT_DOC)
    _write_yaml(tmp_path / 'b_metrics.yaml', CROSS_VAL_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/api/results')
    assert resp.status_code == 200
    by_id = {r['id']: r for r in resp.json()}
    assert by_id['a_metrics']['eval_type'] == 'single_split'
    assert by_id['a_metrics']['model_names'] == ['lstm', 'transformer']
    assert by_id['b_metrics']['eval_type'] == 'cross_validation'
    assert by_id['b_metrics']['stratum'] == 'baseline'


def test_result_detail_passes_models_payload_through_untouched(monkeypatch, tmp_path):
    """The nested 'models' dict is served as-is -- the API doesn't need to know
    the difference between the single-split and cross-validation shapes."""
    _write_yaml(tmp_path / 'cv_metrics.yaml', CROSS_VAL_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/api/results/cv_metrics')
    assert resp.status_code == 200
    body = resp.json()
    assert body['models']['lstm']['by_group']['OA']['rrmse_w_mean'] == 0.12
    assert body['extra']['scheme'] == 'stratified 5-fold'
    assert body['extra']['n_folds'] == 5


# ── Forward / backward compatibility ────────────────────────────────────────────

def test_forward_compat_unexpected_top_level_key_is_preserved(monkeypatch, tmp_path):
    """A brand-new top-level field appearing in a future export (not one of
    today's known envelope keys) should surface in `extra`, not break parsing."""
    doc = dict(SINGLE_SPLIT_DOC)
    doc['confidence_interval_method'] = 'bootstrap'   # hypothetical future field
    _write_yaml(tmp_path / 'future_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/api/results/future_metrics')
    assert resp.status_code == 200
    assert resp.json()['extra']['confidence_interval_method'] == 'bootstrap'


def test_backward_compat_missing_optional_envelope_fields(monkeypatch, tmp_path):
    """A file with no 'stratum' and no 'eval_type' key at all (e.g. a run from
    before those fields existed) should default cleanly, never KeyError/500."""
    doc = {'version': 'Uhlrich_v2.0', 'dataset_version': 'Uhlrich_v2',
           'date': '2026-01-01', 'models': {'lstm': {'test_mse': 0.02}}}
    _write_yaml(tmp_path / 'old_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/api/results/old_metrics')
    assert resp.status_code == 200
    body = resp.json()
    assert body['eval_type'] == 'unknown'
    assert body['stratum'] is None
    assert body['dataset'] is None


# ── Malformed / mixed directory contents ────────────────────────────────────────

def test_scan_skips_malformed_yaml_and_returns_the_rest(monkeypatch, tmp_path):
    """One bad file in the directory shouldn't take down the whole listing."""
    _write_yaml(tmp_path / 'good_metrics.yaml', SINGLE_SPLIT_DOC)
    (tmp_path / 'broken_metrics.yaml').write_text('key: [unclosed', encoding='utf-8')
    (tmp_path / 'list_metrics.yaml').write_text('- just\n- a\n- list\n', encoding='utf-8')
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/api/results')
    assert resp.status_code == 200
    ids = {r['id'] for r in resp.json()}
    assert ids == {'good_metrics'}


def test_unknown_result_id_returns_404(monkeypatch, tmp_path):
    _patch_metrics_dir(monkeypatch, tmp_path)
    resp = client.get('/api/results/does_not_exist')
    assert resp.status_code == 404


# ── Query filters ────────────────────────────────────────────────────────────────

def test_filters_by_dataset_stratum_and_eval_type(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'a_metrics.yaml', SINGLE_SPLIT_DOC)
    _write_yaml(tmp_path / 'b_metrics.yaml', CROSS_VAL_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    assert [r['id'] for r in client.get('/api/results?dataset=uhlrich').json()] == ['a_metrics']
    assert [r['id'] for r in client.get('/api/results?stratum=baseline').json()] == ['b_metrics']
    assert [r['id'] for r in
            client.get('/api/results?eval_type=cross_validation').json()] == ['b_metrics']


# ── Dataset/version endpoints — real, checked-in versions/ dir ─────────────────
# (static data, not runtime output, so it's safe to read the real files here.)

def test_datasets_endpoint_lists_real_datasets_and_versions():
    resp = client.get('/api/datasets')
    assert resp.status_code == 200
    by_name = {d['dataset']: d for d in resp.json()['datasets']}
    assert 'silder' in by_name
    assert 'uhlrich' in by_name
    assert set(by_name['silder']['versions']) >= {'1', '2', '3'}


def test_dataset_version_detail_matches_real_file():
    resp = client.get('/api/datasets/uhlrich/versions/3')
    assert resp.status_code == 200
    body = resp.json()
    assert body['version'] == 'uhlrich_v3'
    assert 'trial_name' in body['description']


def test_dataset_version_unknown_returns_404():
    resp = client.get('/api/datasets/uhlrich/versions/999')
    assert resp.status_code == 404


# ── HTML browsing UI ─────────────────────────────────────────────────────────

def test_html_results_list_empty_state(monkeypatch, tmp_path):
    _patch_metrics_dir(monkeypatch, tmp_path)
    resp = client.get('/')
    assert resp.status_code == 200
    assert 'text/html' in resp.headers['content-type']
    assert 'No results yet' in resp.text


def test_html_results_list_shows_result_link(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'a_metrics.yaml', SINGLE_SPLIT_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)
    resp = client.get('/')
    assert resp.status_code == 200
    assert 'href="/results/a_metrics"' in resp.text
    assert 'single_split' in resp.text


def test_html_results_list_filters_by_query_param(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'a_metrics.yaml', SINGLE_SPLIT_DOC)
    _write_yaml(tmp_path / 'b_metrics.yaml', CROSS_VAL_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)
    resp = client.get('/?dataset=uhlrich')
    assert 'a_metrics' in resp.text
    assert 'b_metrics' not in resp.text


def test_html_result_detail_renders_comparison_table_and_nested_lists(monkeypatch, tmp_path):
    """Regression check: nested list/dict fields (e.g. per-fold arrays, by_group)
    must render as their actual values, not silently break because a dict key
    happens to collide with a builtin dict method name like 'values'/'items'."""
    doc = dict(CROSS_VAL_DOC)
    doc['models'] = {
        'lstm': {
            'rrmse_w_mean': 0.11,
            'per_fold_rrmse_w': [0.1, 0.12, 0.11],
            'by_group': {'OA': {'rrmse_w_mean': 0.12}, 'Y': {'rrmse_w_mean': 0.10}},
        },
    }
    _write_yaml(tmp_path / 'cv_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/cv_metrics')
    assert resp.status_code == 200
    assert 'rrmse_w_mean' in resp.text
    assert '0.1000, 0.1200, 0.1100' in resp.text   # per_fold_rrmse_w, comma-joined
    assert 'OA' in resp.text and '0.1200' in resp.text   # by_group, recursively rendered


def test_html_result_detail_unknown_id_returns_404(monkeypatch, tmp_path):
    _patch_metrics_dir(monkeypatch, tmp_path)
    resp = client.get('/results/does_not_exist')
    assert resp.status_code == 404


# ── units + subset UI clarity ────────────────────────────────────────────────

def test_html_result_detail_shows_subset_and_primary_metric(monkeypatch, tmp_path):
    doc = dict(SINGLE_SPLIT_DOC)
    doc['subset'] = ['knee_fy', 'ankle_fy']
    doc['primary_metric'] = 'rrmse_w'
    _write_yaml(tmp_path / 'a_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/a_metrics')
    assert 'class="subset-details"' in resp.text
    assert '2 of the full output set' in resp.text
    assert 'knee_fy, ankle_fy' in resp.text
    assert '<code>rrmse_w</code>' in resp.text


def test_html_result_detail_no_subset_row_when_absent(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'a_metrics.yaml', SINGLE_SPLIT_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/a_metrics')
    assert 'class="subset-details"' not in resp.text


def test_html_result_detail_per_output_rows_tagged_with_subset(monkeypatch, tmp_path):
    doc = dict(SINGLE_SPLIT_DOC)
    doc['subset'] = [f'm{i}' for i in range(3)]   # first half of the 10 outputs below
    doc['models'] = {
        'lstm': {'per_output_rrmse': {f'm{i}': i / 10 for i in range(10)}},
    }
    _write_yaml(tmp_path / 'a_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/a_metrics')
    assert resp.text.count('class="subset-tag"') == 3


def test_html_result_detail_mae_field_gets_unit_others_dont(monkeypatch, tmp_path):
    doc = dict(SINGLE_SPLIT_DOC)
    doc['models'] = {'lstm': {'mae_overall': 0.5, 'rrmse_w': 0.1}}
    _write_yaml(tmp_path / 'a_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/a_metrics')
    assert '0.5000 N/kg' in resp.text
    assert '0.1000 N/kg' not in resp.text
    assert '>0.1000<' in resp.text


# ── cross_stratum_transfer detail view ──────────────────────────────────────

def test_html_transfer_detail_shows_ratio_definition_callout(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/transfer_metrics')
    assert resp.status_code == 200
    assert 'class="ratio-callout"' in resp.text
    assert 'gasmed force ALONE' in resp.text


def test_html_transfer_detail_shows_direction_table_and_ratio_cards(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/transfer_metrics')
    assert 'Reconstruction accuracy by direction' in resp.text
    assert 'base → ret' in resp.text
    assert 'class="ratio-card"' in resp.text
    assert '0.9600' in resp.text   # lstm_attn base_trained spearman_r


def test_html_transfer_detail_flags_non_significant_condition(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/transfer_metrics')
    assert 'ratio-ns' in resp.text
    assert '<span class="ns-tag">n.s.</span>' in resp.text


def test_html_transfer_detail_offers_live_predictions_link_without_running_inference(
    monkeypatch, tmp_path,
):
    """Live per-subject predictions run real inference (can take up to a
    minute -- see inference.compute_gastroc_soleus_predictions), so the main
    result page must NOT run them -- only the dedicated fragment route
    (fetched via JS, see loadLivePredictions in base.html) does."""
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    def _boom(*a, **kw):
        raise AssertionError('should not run live inference on the main result page')
    monkeypatch.setattr(inference, 'compute_gastroc_soleus_predictions', _boom)

    resp = client.get('/results/transfer_metrics')
    assert resp.status_code == 200
    assert 'href="/results/transfer_metrics/ratio-predictions"' in resp.text
    assert 'onclick="loadLivePredictions(this,' in resp.text
    assert 'class="subject-scroll"' not in resp.text


def _fake_predictions(version, dataset, model_names):
    return {
        'subjects': ['Subject1'],
        'ground_truth': {'Subject1': {'baseline': 0.5, 'retention': 0.4}},
        'predictions': {
            'lstm_attn': {
                'base_trained': {'Subject1': {'baseline': 0.5, 'retention': 0.4}},
                'ret_trained': {'Subject1': {'baseline': 0.5, 'retention': 0.4}},
            },
        },
    }


def test_html_ratio_predictions_fragment_shows_subject_panels(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(inference, 'compute_gastroc_soleus_predictions', _fake_predictions)

    resp = client.get('/results/transfer_metrics/ratio-predictions')
    assert resp.status_code == 200
    assert 'class="subject-scroll"' in resp.text
    assert '<h4>Subject1</h4>' in resp.text
    assert 'class="ratio-legend"' in resp.text
    # Fragment only -- no page chrome from base.html.
    assert '<nav' not in resp.text


def test_html_ratio_predictions_fragment_shows_best_model_legend(monkeypatch, tmp_path):
    """TRANSFER_DOC's lstm_attn has spearman_r=0.96 (base_trained) and 0.01
    (ret_trained) with only one model -- it should be picked for both."""
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(inference, 'compute_gastroc_soleus_predictions', _fake_predictions)

    resp = client.get('/results/transfer_metrics/ratio-predictions')
    assert 'lstm_attn (base-trained)' in resp.text
    assert 'lstm_attn (ret-trained)' in resp.text


def test_html_ratio_predictions_404s_for_non_transfer_result(monkeypatch, tmp_path):
    _write_yaml(tmp_path / 'single_metrics.yaml', SINGLE_SPLIT_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/single_metrics/ratio-predictions')
    assert resp.status_code == 404


def test_html_transfer_detail_survives_model_without_ratio_recovery(monkeypatch, tmp_path):
    """cnn_lstm in TRANSFER_DOC has no ratio_recovery block at all (transfer
    not yet run for that arch) -- must not 500, must just not get a card."""
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/transfer_metrics')
    assert resp.status_code == 200
    assert resp.text.count('class="ratio-card"') == 1   # only lstm_attn


def test_html_transfer_detail_uses_generic_comparison_for_other_eval_types(
    monkeypatch, tmp_path,
):
    """Sanity check the branch: a non-transfer doc still goes through the
    normal _model_comparison.html path, not the transfer one."""
    _write_yaml(tmp_path / 'single_metrics.yaml', SINGLE_SPLIT_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/single_metrics')
    assert 'class="ratio-card"' not in resp.text
    assert 'Model comparison' in resp.text




def test_html_results_list_handles_transfer_eval_type_without_crashing(
    monkeypatch, tmp_path,
):
    """models[name] being direction-keyed (all-dict values, no scalars) must
    degrade to the muted model-names fallback on the list page, not crash
    pick_primary_metric/bar_chart_svg."""
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/')
    assert resp.status_code == 200
    assert 'pill-cross_stratum_transfer' in resp.text


# ── /metrics glossary ────────────────────────────────────────────────────────

def test_metrics_glossary_covers_key_terms():
    resp = client.get('/metrics')
    assert resp.status_code == 200
    for term in ('Spearman', 'spread ratio', 'Bland-Altman', 'RRMSE', 'Dice'):
        assert term in resp.text


def test_metrics_glossary_covers_auc_addmagmid_and_gasmed_notes():
    resp = client.get('/metrics')
    assert 'id="auc"' in resp.text
    assert 'addmagMid' in resp.text
    assert 'gasmed-only' in resp.text
    assert 'Why both R' in resp.text


def test_model_comparison_and_transfer_views_link_to_metrics_glossary(
    monkeypatch, tmp_path,
):
    _write_yaml(tmp_path / 'single_metrics.yaml', SINGLE_SPLIT_DOC)
    _write_yaml(tmp_path / 'transfer_metrics.yaml', TRANSFER_DOC)
    _patch_metrics_dir(monkeypatch, tmp_path)

    assert 'href="/metrics"' in client.get('/results/single_metrics').text
    assert 'href="/metrics#ratio-recovery"' in client.get('/results/transfer_metrics').text


def test_html_datasets_list_shows_real_datasets():
    resp = client.get('/datasets')
    assert resp.status_code == 200
    assert 'uhlrich' in resp.text
    assert 'href="/datasets/silder/versions/3"' in resp.text


def test_html_dataset_version_detail_renders_description():
    resp = client.get('/datasets/uhlrich/versions/3')
    assert resp.status_code == 200
    assert 'trial_name' in resp.text


def test_html_dataset_version_unknown_returns_404():
    resp = client.get('/datasets/uhlrich/versions/999')
    assert resp.status_code == 404


def test_html_result_detail_large_per_output_field_gets_own_chart_not_generic_dump(
    monkeypatch, tmp_path,
):
    """A large flat metric map (e.g. per_output_rrmse) should render via the
    filterable per-output chart, and NOT also appear in the generic recursive
    'additional fields' dump (that would just be the mind-numbing table twice)."""
    doc = dict(SINGLE_SPLIT_DOC)
    doc['models'] = {
        'lstm': {'rrmse_w': 0.1, 'per_output_rrmse': {f'muscle{i}': i / 10 for i in range(12)}},
    }
    _write_yaml(tmp_path / 'big_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/big_metrics')
    assert resp.status_code == 200
    assert '<option value="output-panel-0">per_output_rrmse</option>' in resp.text
    assert 'class="output-filter"' in resp.text
    # 'muscle0' legitimately appears twice per row (data-key attr + label text) --
    # anything more would mean it also leaked into a second, generic dump.
    assert resp.text.count('muscle0') == 2
    assert 'additional fields' not in resp.text


def test_html_result_detail_per_output_dropdown_lists_all_metric_fields(
    monkeypatch, tmp_path,
):
    """Multiple per-output fields (rrmse/mae/r2/dice) each get one dropdown
    option and one panel, with only the first panel visible by default --
    switching is client-side JS, not a server round trip."""
    outputs = {f'muscle{i}': i / 10 for i in range(10)}
    doc = dict(SINGLE_SPLIT_DOC)
    doc['models'] = {
        'lstm': {
            'per_output_rrmse': dict(outputs),
            'per_output_mae': dict(outputs),
            'per_output_r2': dict(outputs),
        },
    }
    _write_yaml(tmp_path / 'multi_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/multi_metrics')
    assert resp.status_code == 200
    for field in ('per_output_rrmse', 'per_output_mae', 'per_output_r2'):
        assert f'>{field}</option>' in resp.text
    assert resp.text.count('class="output-panel"') == 3
    # Only the first of 3 panels starts visible -- the other two are hidden
    # until the dropdown switches them in via JS. (The pinned-charts container
    # also starts hidden, so match each panel's own opening tag specifically
    # rather than counting every 'display:none' on the page.)
    panel_tags = re.findall(r'<div class="output-panel" id="output-panel-\d+"[^>]*>', resp.text)
    assert len(panel_tags) == 3
    assert sum('display:none' in tag for tag in panel_tags) == 2


def test_html_result_detail_per_output_bar_value_is_visible_not_hover_only(
    monkeypatch, tmp_path,
):
    doc = dict(SINGLE_SPLIT_DOC)
    doc['models'] = {'lstm': {'per_output_rrmse': {f'm{i}': 0.1234 for i in range(10)}}}
    _write_yaml(tmp_path / 'visible_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/visible_metrics')
    assert '<span class="output-bar-value">0.1234</span>' in resp.text


def test_html_result_detail_has_pin_button_and_pinned_container(monkeypatch, tmp_path):
    """The 'pin this view' affordance is client-side JS (cloning the current
    panel's DOM) -- this just checks the markup/wiring it depends on is
    present: the button, its onclick target, and the container it appends to."""
    doc = dict(SINGLE_SPLIT_DOC)
    doc['models'] = {'lstm': {'per_output_rrmse': {f'm{i}': 0.1 for i in range(10)}}}
    _write_yaml(tmp_path / 'pin_metrics.yaml', doc)
    _patch_metrics_dir(monkeypatch, tmp_path)

    resp = client.get('/results/pin_metrics')
    assert 'onclick="saveOutputChart(this)"' in resp.text
    assert 'id="pinned-output-charts"' in resp.text
    assert 'id="pinned-output-grid"' in resp.text
    assert 'function saveOutputChart(button)' in resp.text


# ── /predict ─────────────────────────────────────────────────────────────────

def _predict_fixture_paths(tmp_path, monkeypatch):
    paths = {
        'model_dir': tmp_path / 'models',
        'splits_dir': tmp_path / 'splits',
        'test_data_dir': tmp_path / 'test_data',
    }
    for p in paths.values():
        p.mkdir()
    monkeypatch.setattr(inference, '_paths', lambda: paths)
    return paths


def _write_predict_fixture(paths, n_inputs=5, n_outputs=3, n_samples=4, seq_len=10,
                           output_keys=None):
    bp = {'hidden_size': 4, 'num_layers': 1, 'dropout_rate': 0.0}
    model = build_model('lstm', bp, n_inputs, n_outputs, torch.device('cpu'))
    torch.save(model.state_dict(), paths['model_dir'] / 'Test_v1.0_lstm.pth')

    study = optuna.create_study(study_name='Test_v1.0_lstm',
                                storage=f'sqlite:///{paths["model_dir"] / "optuna_v1.db"}')
    study.add_trial(create_trial(
        params=bp,
        distributions={k: CategoricalDistribution([v]) for k, v in bp.items()},
        value=0.0,
    ))

    rng = np.random.default_rng(0)
    X = rng.random((n_samples, seq_len, n_inputs)).astype(np.float32)
    y = rng.random((n_samples, seq_len, n_outputs)).astype(np.float32)
    if output_keys is None:
        output_keys = [f'out{i}' for i in range(n_outputs)]
    np.savez(paths['splits_dir'] / 'Test_v1_test_data.npz',
            X_test=X, y_test=y, output_keys=np.array(output_keys))


def _write_second_model_fixture(paths, n_inputs=5, n_outputs=3):
    """A second, differently-architected model against the SAME npz written
    by _write_predict_fixture, for multi-run comparison tests."""
    bp = {'cnn_channels': 4, 'hidden_size': 4, 'num_layers': 1, 'dropout_rate': 0.0}
    model = build_model('cnn_lstm', bp, n_inputs, n_outputs, torch.device('cpu'))
    torch.save(model.state_dict(), paths['model_dir'] / 'Test_v1.0_cnn_lstm.pth')
    study = optuna.create_study(study_name='Test_v1.0_cnn_lstm',
                                storage=f'sqlite:///{paths["model_dir"] / "optuna_v1.db"}')
    study.add_trial(create_trial(
        params=bp,
        distributions={k: CategoricalDistribution([v]) for k, v in bp.items()},
        value=0.0,
    ))


def test_predict_page_loads_with_no_selection(monkeypatch, tmp_path):
    _predict_fixture_paths(tmp_path, monkeypatch)
    resp = client.get('/predict')
    assert resp.status_code == 200
    assert 'Pick a model and test-data file' in resp.text


def test_predict_end_to_end_shows_waveform_and_stats(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz',
    })
    assert resp.status_code == 200
    assert resp.text.count('<polyline') == 2   # ground truth + prediction
    assert 'rrmse_w' in resp.text
    assert 'auc_mean' in resp.text
    assert 'sample 0 of 3' in resp.text


def test_predict_no_shaded_region_for_output_outside_reported_subset(monkeypatch, tmp_path):
    """The fixture's synthetic output names (out0/out1/out2) aren't part of
    any config.yaml reporting subset -- the overlap shading must stay off,
    with the explanatory note, rather than shading an unheadlined output."""
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz', 'output_key': 'out0',
    })
    assert '<polygon' not in resp.text
    assert "isn't part of the standard reported subset" in resp.text


def test_predict_shows_shaded_region_for_output_inside_reported_subset(monkeypatch, tmp_path):
    """knee_fy is a real reporting.V1_outputs member -- picking it should
    turn the overlap shading on."""
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, n_outputs=3,
                           output_keys=['knee_fy', 'addmagMid', 'ankle_fy'])

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz', 'output_key': 'knee_fy',
    })
    assert '<polygon' in resp.text
    assert 'shaded region — overlap between ground truth and prediction' in resp.text


def test_predict_switching_to_non_subset_output_turns_shading_off(monkeypatch, tmp_path):
    """Same run, different output_key -- addmagMid is deliberately not in any
    reporting subset (see the glossary's addmagMid example)."""
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, n_outputs=3,
                           output_keys=['knee_fy', 'addmagMid', 'ankle_fy'])

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz', 'output_key': 'addmagMid',
    })
    assert '<polygon' not in resp.text


def test_predict_output_dropdown_grouped_by_subset_once_pth_selected(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, n_outputs=3,
                           output_keys=['knee_fy', 'addmagMid', 'ankle_fy'])

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz',
    })
    assert resp.text.count('<optgroup') == 2
    reported = resp.text.split('Reported subset')[1].split('</optgroup>')[0]
    all_outputs = resp.text.split('All outputs')[1].split('</optgroup>')[0]
    assert 'knee_fy' in reported and 'ankle_fy' in reported
    assert 'addmagMid' not in reported
    assert 'addmagMid' in all_outputs


def test_predict_output_dropdown_ungrouped_without_pth(monkeypatch, tmp_path):
    """Subset membership needs a .pth's dataset version to resolve -- with
    only npz picked so far, the dropdown falls back to one flat list."""
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, output_keys=['knee_fy', 'addmagMid', 'ankle_fy'])

    resp = client.get('/predict', params={'npz': 'Test_v1_test_data.npz'})
    assert '<optgroup' not in resp.text
    assert 'knee_fy' in resp.text


def test_predict_sample_navigation_via_query_param(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, n_samples=4)

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz', 'sample_idx': 2,
    })
    assert 'sample 2 of 3' in resp.text
    # Neither nav link should be disabled in the middle of the range.
    assert 'class="disabled"' not in resp.text.split('predict-nav')[1].split('</div>')[0]


def test_predict_mismatched_shapes_shows_error_not_500(monkeypatch, tmp_path):
    """A model/test-data pick with incompatible output counts must fail with
    a readable error, not a raw 500 -- this is a very pickable-by-accident
    combination given both dropdowns are just flat lists of filenames."""
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, n_outputs=3)
    # Overwrite the npz with a mismatched output count, same filename.
    rng = np.random.default_rng(1)
    X = rng.random((4, 10, 5)).astype(np.float32)
    y = rng.random((4, 10, 9)).astype(np.float32)
    np.savez(paths['splits_dir'] / 'Test_v1_test_data.npz',
            X_test=X, y_test=y, output_keys=np.array([f'o{i}' for i in range(9)]))

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz',
    })
    assert resp.status_code == 200
    assert 'class="predict-error"' in resp.text
    assert '<polyline' not in resp.text


def test_predict_unknown_pth_shows_error(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params={
        'pth': 'does_not_exist.pth', 'npz': 'Test_v1_test_data.npz',
    })
    assert resp.status_code == 200
    assert 'class="predict-error"' in resp.text


# ── /predict multi-run comparison ("pin model/data sets") ──────────────────────

def test_predict_add_run_href_offered_for_unpinned_selection(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz',
    })
    assert 'add this model/data to comparison' in resp.text
    assert 'runs=Test_v1.0_lstm.pth%3A%3ATest_v1_test_data.npz' in resp.text


def test_predict_add_run_href_absent_once_already_pinned(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params={
        'pth': 'Test_v1.0_lstm.pth', 'npz': 'Test_v1_test_data.npz',
        'runs': 'Test_v1.0_lstm.pth::Test_v1_test_data.npz',
    })
    assert 'add this model/data to comparison' not in resp.text


def test_predict_two_pinned_runs_show_combined_comparison_table(monkeypatch, tmp_path):
    # n_outputs=9 (>= build_per_output_charts' min_items=8) so the per-output
    # section actually renders -- a real trained model has 40+ outputs, this
    # just needs to clear that threshold, not match it exactly.
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, n_outputs=9)
    _write_second_model_fixture(paths, n_outputs=9)

    resp = client.get('/predict', params=[
        ('pth', 'Test_v1.0_lstm.pth'), ('npz', 'Test_v1_test_data.npz'),
        ('runs', 'Test_v1.0_lstm.pth::Test_v1_test_data.npz'),
        ('runs', 'Test_v1.0_cnn_lstm.pth::Test_v1_test_data.npz'),
    ])
    assert resp.status_code == 200
    assert 'Test_v1.0_lstm · Test_v1_test_data' in resp.text
    assert 'Test_v1.0_cnn_lstm · Test_v1_test_data' in resp.text
    assert '<h2>Model comparison</h2>' in resp.text
    assert '<h2>Per-output metrics</h2>' in resp.text
    assert 'class="predict-error"' not in resp.text


def test_predict_remove_run_link_strips_only_that_entry(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)
    _write_second_model_fixture(paths)

    resp = client.get('/predict', params=[
        ('runs', 'Test_v1.0_lstm.pth::Test_v1_test_data.npz'),
        ('runs', 'Test_v1.0_cnn_lstm.pth::Test_v1_test_data.npz'),
    ])
    remove_hrefs = re.findall(r'href="(/predict\?[^"]*)" title="remove"', resp.text)
    assert len(remove_hrefs) == 2
    # Removing the first run's chip should leave exactly the second one pinned.
    remaining = client.get(remove_hrefs[0].replace('&amp;', '&'))
    assert 'Test_v1.0_cnn_lstm · Test_v1_test_data' in remaining.text
    assert 'Test_v1.0_lstm · Test_v1_test_data' not in remaining.text


def test_predict_duplicate_pinned_run_deduplicates(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params=[
        ('runs', 'Test_v1.0_lstm.pth::Test_v1_test_data.npz'),
        ('runs', 'Test_v1.0_lstm.pth::Test_v1_test_data.npz'),
    ])
    assert resp.text.count('class="run-chip"') == 1


def test_predict_malformed_pinned_run_entry_shows_error_not_500(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params=[('runs', 'not-a-valid-entry-no-separator')])
    assert resp.status_code == 200
    assert 'class="predict-error"' in resp.text


def test_predict_pinned_run_with_missing_pth_shows_error_not_500(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths)

    resp = client.get('/predict', params=[
        ('runs', 'does_not_exist.pth::Test_v1_test_data.npz'),
    ])
    assert resp.status_code == 200
    assert 'class="predict-error"' in resp.text
    assert 'does_not_exist.pth' in resp.text


def test_predict_nav_links_preserve_pinned_runs(monkeypatch, tmp_path):
    paths = _predict_fixture_paths(tmp_path, monkeypatch)
    _write_predict_fixture(paths, n_samples=4)

    resp = client.get('/predict', params=[
        ('pth', 'Test_v1.0_lstm.pth'), ('npz', 'Test_v1_test_data.npz'), ('sample_idx', 1),
        ('runs', 'Test_v1.0_lstm.pth::Test_v1_test_data.npz'),
    ])
    next_href = re.search(r'href="([^"]*)">next', resp.text).group(1).replace('&amp;', '&')
    assert 'runs=Test_v1.0_lstm.pth' in next_href


# ── /about ───────────────────────────────────────────────────────────────────
# README.md is static, checked-in content -- like the versions/ dir tests
# above, safe to read the real file directly rather than needing a fixture.

def test_about_page_renders_real_readme_content():
    resp = client.get('/about')
    assert resp.status_code == 200
    assert '<h1>GRF Muscle Prediction</h1>' in resp.text
    assert 'Problem Statement' in resp.text


def test_about_page_includes_background_section():
    resp = client.get('/about')
    assert 'about-background' in resp.text
    assert 'static optimization' in resp.text.lower()


def test_about_page_survives_missing_readme(monkeypatch):
    """A missing README shouldn't 500 the page -- the hand-written background
    section should still render even with an empty readme_html."""
    monkeypatch.setattr(data_access, 'README_PATH', data_access.README_PATH.parent / 'nope.md')
    resp = client.get('/about')
    assert resp.status_code == 200
    assert 'about-background' in resp.text


def test_get_readme_html_converts_markdown_headings(tmp_path, monkeypatch):
    readme = tmp_path / 'README.md'
    readme.write_text('# Title\n\nSome *text*.\n', encoding='utf-8')
    monkeypatch.setattr(data_access, 'README_PATH', readme)
    html = data_access.get_readme_html()
    assert '<h1>Title</h1>' in html
    assert '<em>text</em>' in html


def test_get_readme_html_empty_string_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_access, 'README_PATH', tmp_path / 'nope.md')
    assert data_access.get_readme_html() == ''


# ── theme: no emoji ──────────────────────────────────────────────────────────

_EMOJI_RE = re.compile('[\U0001F300-\U0001FAFF\U00002600-\U000027BF]')


def test_no_emoji_on_key_pages(monkeypatch, tmp_path):
    """Explicit ask: the theme dropped emoji entirely -- guard against a
    future button/label accidentally reintroducing one."""
    _predict_fixture_paths(tmp_path, monkeypatch)
    for path in ('/', '/predict', '/about', '/datasets'):
        resp = client.get(path)
        assert not _EMOJI_RE.search(resp.text), f'{path} contains an emoji character'
