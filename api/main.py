from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api import data_access, inference, render
from api.schemas import DatasetsResponse, DatasetVersionInfo, ResultDetail, ResultSummary
from grf_pipeline_utils.eval_utils import (
    calc_auc_overall,
    calc_mae_overall,
    calc_r2_per_output,
    calc_rrmse_weighted,
)

app = FastAPI(
    title='GRF Muscle Prediction — Results API',
    description=(
        'Browse evaluation results (single train/test splits and cross-validation '
        'runs) across datasets, dataset versions, and strata. Schema-light by '
        'design: only a small envelope (dataset/version/stratum/eval_type) is '
        'strictly typed -- per-model metrics pass through as-is, since the '
        'underlying metric framework is still evolving.'
    ),
    version='0.1.0',
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / 'templates'))
templates.env.filters['fmt'] = render.fmt_value
templates.env.filters['fmt_unit'] = render.fmt_with_unit
templates.env.filters['scalarlist'] = render.scalar_list_or_none


# ── JSON API ─────────────────────────────────────────────────────────────────

@app.get('/api/results', response_model=list[ResultSummary])
def get_results(dataset: Optional[str] = None, stratum: Optional[str] = None,
                eval_type: Optional[str] = None):
    results = data_access.list_results()
    if dataset is not None:
        results = [r for r in results if r['dataset'] == dataset]
    if stratum is not None:
        results = [r for r in results if r['stratum'] == stratum]
    if eval_type is not None:
        results = [r for r in results if r['eval_type'] == eval_type]
    return results


@app.get('/api/results/{result_id}', response_model=ResultDetail)
def get_result(result_id: str):
    result = data_access.get_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f'No result found for id {result_id!r}')
    return result


@app.get('/api/datasets', response_model=DatasetsResponse)
def get_datasets():
    return data_access.list_datasets()


@app.get('/api/datasets/{dataset}/versions/{version}', response_model=DatasetVersionInfo)
def get_dataset_version(dataset: str, version: str):
    info = data_access.get_dataset_version(dataset, version)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f'No version doc found for dataset={dataset!r} version={version!r}',
        )
    return info


# ── HTML browsing UI ─────────────────────────────────────────────────────────
# Reads the same data_access functions as the JSON API above -- these routes
# exist purely to make that data human-scannable without going through /docs.

def _build_comparison_context(models: dict[str, dict], extra: dict[str, Any] = None,
                              declared_primary_metric: Optional[str] = None,
                              subset: Optional[list[str]] = None) -> dict:
    """Shared by html_result_detail (models from a saved metrics.yaml) and
    html_predict (models from live inference.summarize_run calls) -- same
    shape in, same table/per-output-chart context out. `declared_primary_metric`
    is the envelope's own top-level 'primary_metric' field (data_access.py
    promotes it out of 'extra'), not something to re-derive from extra here.
    `subset` (the result's own reported-subset list) tags per-output rows --
    left unset for html_predict's pinned-comparison table, where multiple
    runs could span different dataset versions with no single subset."""
    extra = extra or {}
    metric = render.pick_primary_metric(models, declared_primary_metric)
    reference = render.infer_reference(metric, extra) if metric else None

    per_output_charts = render.build_per_output_charts(models, subset=subset)
    exclude = frozenset(c['field_name'] for c in per_output_charts)
    table = render.build_models_table(models, exclude_keys=exclude)
    for row in table['rows']:
        row['reference'] = render.infer_reference(row['metric'], extra)

    return {
        'table': table,
        'primary_metric': metric,
        'reference': reference,
        'chart_svg': render.bar_chart_svg(
            models, metric, reference_value=reference['value'] if reference else None,
        ) if metric else '',
        'per_output_charts': per_output_charts,
    }


@app.get('/', response_class=HTMLResponse, include_in_schema=False)
def html_results_list(request: Request, dataset: Optional[str] = None,
                      stratum: Optional[str] = None, eval_type: Optional[str] = None):
    results = get_results(dataset, stratum, eval_type)
    cards = []
    for r in results:
        metric = render.pick_primary_metric(r['models'], r.get('primary_metric'))
        cards.append({
            **r,
            'primary_metric': metric,
            'metric_caption': render.describe_primary_metric(
                metric, r.get('eval_type'), r.get('subset'),
            ),
            'chart_svg': render.bar_chart_svg(
                r['models'], metric, width=300, bar_height=16, gap=5, label_width=64,
            ) if metric else '',
        })
    return templates.TemplateResponse(request, 'results_list.html', {
        'results': cards,
        'filters': {'dataset': dataset, 'stratum': stratum, 'eval_type': eval_type},
    })


@app.get('/results/{result_id}', response_class=HTMLResponse, include_in_schema=False)
def html_result_detail(request: Request, result_id: str):
    result = data_access.get_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f'No result found for id {result_id!r}')

    # cross_stratum_transfer's models are keyed by direction (base_base/
    # base_ret/ret_ret/ret_base) plus a ratio_recovery block -- structurally
    # different from the flat {metric: value} shape _build_comparison_context
    # assumes, so it gets its own builder + template partial instead.
    if result['eval_type'] == 'cross_stratum_transfer':
        return templates.TemplateResponse(request, 'result_detail.html', {
            'result': result,
            'transfer_comparison': render.build_transfer_comparison(result['models']),
            'ratio_definition': result['extra'].get('ratio_definition'),
            'ratio_predictions_href': f'/results/{result_id}/ratio-predictions',
        })

    return templates.TemplateResponse(request, 'result_detail.html', {
        'result': result,
        **_build_comparison_context(
            result['models'], result['extra'], result.get('primary_metric'),
            result.get('subset'),
        ),
    })


@app.get('/results/{result_id}/ratio-predictions', response_class=HTMLResponse,
         include_in_schema=False)
def html_ratio_predictions(request: Request, result_id: str):
    """Fragment-only response (no base.html chrome) for the transfer page's
    live per-subject prediction panels. Fetched via JS (loadLivePredictions
    in base.html) so clicking the link shows an immediate loading message
    instead of a silently-blank page for the ~minute this can take; the
    <a href> also points straight here as a no-JS fallback (functional but
    unstyled, since a fragment response skips base.html's <style> block)."""
    result = data_access.get_result(result_id)
    if result is None or result['eval_type'] != 'cross_stratum_transfer':
        raise HTTPException(
            status_code=404, detail=f'No transfer result found for id {result_id!r}',
        )

    transfer_comparison = render.build_transfer_comparison(result['models'])
    predictions_data = inference.compute_gastroc_soleus_predictions(
        result['version'] or '', result['dataset'] or '', sorted(result['models'].keys()),
    )
    panel_data = render.build_ratio_prediction_panels(
        predictions_data, transfer_comparison['ratio_cards'], transfer_comparison['model_names'],
    )
    return templates.TemplateResponse(request, '_ratio_prediction_panels.html', {
        'ratio_prediction_panels': panel_data['panels'],
        'best_models': panel_data['best_models'],
    })


def _predict_url(pth: Optional[str], npz: Optional[str], output_key: Optional[str],
                 sample_idx: int, runs: list[str]) -> str:
    params: list[tuple[str, Any]] = []
    if pth:
        params.append(('pth', pth))
    if npz:
        params.append(('npz', npz))
    if output_key:
        params.append(('output_key', output_key))
    if sample_idx:
        params.append(('sample_idx', sample_idx))
    params += [('runs', r) for r in runs]
    return '/predict' + (f'?{urlencode(params)}' if params else '')


@app.get('/predict', response_class=HTMLResponse, include_in_schema=False)
def html_predict(request: Request, pth: Optional[str] = None, npz: Optional[str] = None,
                 output_key: Optional[str] = None, sample_idx: int = 0,
                 runs: list[str] = Query(default=[])):
    pth_files = inference.list_pth_files()
    npz_files = inference.list_test_data_files()

    output_keys: list[str] = []
    error: Optional[str] = None
    result_ctx = None

    # Pinned comparison set -- each entry is "pth::npz"; order-preserving
    # dedup so re-adding the same pair via a stale link is a no-op. Computed
    # up front so prev/next sample links below can carry it along too.
    seen_runs: list[str] = []
    for r in runs:
        if r not in seen_runs:
            seen_runs.append(r)

    if npz:
        try:
            output_keys = inference.peek_output_keys(npz)
        except (FileNotFoundError, OSError) as e:
            error = str(e)

    if pth and npz and not error:
        try:
            run = inference.run_on_test_data(pth, npz)
            output_keys = run['output_keys']
            chosen_output = output_key if output_key in output_keys else output_keys[0]
            out_idx = output_keys.index(chosen_output)
            idx = max(0, min(sample_idx, run['n_samples'] - 1))

            y_true_all, y_pred_all = run['y_true'], run['y_pred']
            model_label = pth.removesuffix('.pth')
            waveform_series = {
                'Ground truth': y_true_all[idx, :, out_idx].tolist(),
                model_label: y_pred_all[idx, :, out_idx].tolist(),
            }
            # AUC/integral-agreement is only reported for the standard subset
            # (see config.yaml reporting.V{major}_outputs/primary_outputs) --
            # the shaded overlap visual is scoped the same way, so it doesn't
            # imply a claim about outputs that aren't actually headlined.
            in_subset = inference.is_in_reported_subset(pth, chosen_output)
            result_ctx = {
                'chosen_output': chosen_output,
                'sample_idx': idx,
                'n_samples': run['n_samples'],
                'in_subset': in_subset,
                'chart_svg': render.waveform_svg(
                    waveform_series,
                    fill_between=('Ground truth', model_label) if in_subset else None,
                ),
                'stats': {
                    'rrmse_w': float(calc_rrmse_weighted(y_true_all, y_pred_all)),
                    'r2_mean': float(
                        calc_r2_per_output(y_true_all, y_pred_all, verbose=False).mean()),
                    'auc_mean': float(calc_auc_overall(y_true_all, y_pred_all)),
                    'mae_overall': float(calc_mae_overall(y_true_all, y_pred_all)),
                },
                'prev_href': (_predict_url(pth, npz, chosen_output, idx - 1, seen_runs)
                             if idx > 0 else None),
                'next_href': (_predict_url(pth, npz, chosen_output, idx + 1, seen_runs)
                             if idx < run['n_samples'] - 1 else None),
            }
        # Broad on purpose: mismatched model/test-data picks (e.g. a v1 model
        # against a v3 test file with a different output count) can fail deep
        # inside torch/optuna in ways not worth enumerating -- surface
        # whatever it is as a clean form error instead of a raw 500.
        except Exception as e:
            error = f'{type(e).__name__}: {e}'

    def _run_label(run_pth: str, run_npz: str) -> str:
        return f"{run_pth.removesuffix('.pth')} · {run_npz.removesuffix('.npz')}"

    comparison_models: dict[str, dict] = {}
    run_errors: list[str] = []
    run_links = []
    for r in seen_runs:
        run_pth, _, run_npz = r.partition('::')
        label = _run_label(run_pth, run_npz) if run_npz else r
        run_links.append({
            'label': label,
            'raw': r,
            'remove_href': _predict_url(pth, npz, output_key, sample_idx,
                                        [x for x in seen_runs if x != r]),
        })
        if not run_npz:
            run_errors.append(f'{r}: malformed run entry')
            continue
        try:
            comparison_models[label] = inference.summarize_run(
                inference.run_on_test_data(run_pth, run_npz))
        except Exception as e:
            run_errors.append(f'{label}: {type(e).__name__}: {e}')

    # Always built, even when empty -- render.build_models_table/
    # build_per_output_charts both handle {} cleanly, and _model_comparison
    # .html (shared with result_detail.html) needs these keys defined rather
    # than absent, since Jinja's default Undefined raises on attribute
    # access (table.rows) even inside an {% if %} guard.
    comparison_ctx = _build_comparison_context(comparison_models)

    add_run_href = None
    if pth and npz and f'{pth}::{npz}' not in seen_runs:
        add_run_href = _predict_url(pth, npz, output_key, sample_idx,
                                    [*seen_runs, f'{pth}::{npz}'])

    # Groups the output dropdown by reported-subset membership -- only
    # resolvable once a specific .pth's dataset version is known, so a flat
    # (ungrouped) list is the fallback when only npz is picked so far.
    subset_output_keys = None
    if pth and output_keys:
        subset_output_keys = [k for k in output_keys if inference.is_in_reported_subset(pth, k)]

    return templates.TemplateResponse(request, 'predict.html', {
        'pth_files': pth_files,
        'npz_files': npz_files,
        'output_keys': output_keys,
        'subset_output_keys': subset_output_keys,
        'selected': {'pth': pth, 'npz': npz, 'output_key': output_key, 'sample_idx': sample_idx},
        'error': error,
        'result': result_ctx,
        'run_links': run_links,
        'run_errors': run_errors,
        'add_run_href': add_run_href,
        **comparison_ctx,
    })


@app.get('/about', response_class=HTMLResponse, include_in_schema=False)
def html_about(request: Request):
    return templates.TemplateResponse(request, 'about.html', {
        'readme_html': data_access.get_readme_html(),
    })


@app.get('/metrics', response_class=HTMLResponse, include_in_schema=False)
def html_metrics_glossary(request: Request):
    return templates.TemplateResponse(request, 'metrics_glossary.html', {})


@app.get('/datasets', response_class=HTMLResponse, include_in_schema=False)
def html_datasets_list(request: Request):
    return templates.TemplateResponse(request, 'datasets_list.html', {
        'datasets': data_access.list_datasets()['datasets'],
    })


@app.get('/datasets/{dataset}/versions/{version}', response_class=HTMLResponse,
        include_in_schema=False)
def html_dataset_version_detail(request: Request, dataset: str, version: str):
    info = data_access.get_dataset_version(dataset, version)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f'No version doc found for dataset={dataset!r} version={version!r}',
        )
    return templates.TemplateResponse(request, 'dataset_version_detail.html', {'info': info})
