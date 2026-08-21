from api.render import (
    bar_chart_svg,
    build_models_table,
    build_per_output_charts,
    build_ratio_prediction_panels,
    build_transfer_comparison,
    describe_primary_metric,
    fmt_value,
    fmt_with_unit,
    infer_reference,
    model_color,
    pick_primary_metric,
    scalar_list_or_none,
    slope_chart_svg,
    waveform_svg,
)

# ── fmt_value ────────────────────────────────────────────────────────────────

def test_fmt_value_none_is_em_dash():
    assert fmt_value(None) == '—'


def test_fmt_value_rounds_floats():
    assert fmt_value(0.123456) == '0.1235'


def test_fmt_value_passes_through_strings_and_ints():
    assert fmt_value('baseline') == 'baseline'
    assert fmt_value(7) == '7'


# ── fmt_with_unit ─────────────────────────────────────────────────────────────
# MAE is the one reconstruction-accuracy metric still in the signal's native
# force units (N/kg) -- every other metric (RRMSE/R2/Dice/AUC) is a
# normalized ratio, so it's dimensionless and must NOT get a unit suffix.

def test_fmt_with_unit_appends_unit_for_mae_field_names():
    for key in ('mae', 'mae_overall', 'mae_mean', 'mae_std', 'per_output_mae', 'threshold'):
        assert fmt_with_unit(0.5, key) == '0.5000 N/kg'


def test_fmt_with_unit_no_unit_for_non_mae_metrics():
    for key in ('rrmse_w', 'r2_mean', 'dice_mean', 'auc_mean', 'test_mse'):
        assert fmt_with_unit(0.5, key) == '0.5000'


def test_fmt_with_unit_no_unit_for_mae_derived_ratios():
    """mae_over_threshold_* is MAE / threshold -- dimensionless, not N/kg,
    even though the key contains 'mae'."""
    assert fmt_with_unit(0.2, 'mae_over_threshold_mean') == '0.2000'
    assert fmt_with_unit(0.2, 'mae_over_threshold_std') == '0.2000'


def test_fmt_with_unit_no_unit_without_a_key():
    assert fmt_with_unit(0.5) == '0.5000'
    assert fmt_with_unit(0.5, None) == '0.5000'


def test_fmt_with_unit_no_unit_for_non_numeric_or_none():
    assert fmt_with_unit(None, 'mae') == '—'
    assert fmt_with_unit('n/a', 'mae') == 'n/a'


# ── scalar_list_or_none ──────────────────────────────────────────────────────

def test_scalar_list_or_none_joins_plain_scalars():
    assert scalar_list_or_none([0.1, 0.2, 0.3]) == '0.1000, 0.2000, 0.3000'


def test_scalar_list_or_none_returns_none_for_nested_content():
    assert scalar_list_or_none([{'a': 1}, {'b': 2}]) is None


# ── build_models_table ───────────────────────────────────────────────────────

def test_build_models_table_separates_scalar_from_nested():
    table = build_models_table({
        'lstm': {'rrmse_w': 0.1, 'by_group': {'OA': 1}},
        'cnn_lstm': {'rrmse_w': 0.2, 'per_fold': [1, 2]},
    })
    assert table['model_names'] == ['cnn_lstm', 'lstm']
    metric_names = {row['metric'] for row in table['rows']}
    assert metric_names == {'rrmse_w'}
    assert table['rows'][0]['by_model'] == {'cnn_lstm': 0.2, 'lstm': 0.1}
    assert table['nested']['lstm'] == {'by_group': {'OA': 1}}
    assert table['nested']['cnn_lstm'] == {'per_fold': [1, 2]}


def test_build_models_table_missing_key_in_one_model_is_none_not_dropped():
    """A metric present in only one model's dict should still get a row, with
    the other model showing None rather than the row disappearing."""
    table = build_models_table({'a': {'rrmse_w': 0.1}, 'b': {}})
    row = next(r for r in table['rows'] if r['metric'] == 'rrmse_w')
    assert row['by_model'] == {'a': 0.1, 'b': None}


def test_build_models_table_exclude_keys_drops_from_scalar_and_nested():
    table = build_models_table(
        {'a': {'rrmse_w': 0.1, 'per_output_rrmse': {'x': 1}}},
        exclude_keys=frozenset({'per_output_rrmse'}),
    )
    assert {r['metric'] for r in table['rows']} == {'rrmse_w'}
    assert table['nested']['a'] == {}


# ── model_color ──────────────────────────────────────────────────────────────

def test_model_color_is_stable_regardless_of_input_order():
    names = ['transformer', 'lstm', 'cnn_lstm']
    c1 = model_color('lstm', names)
    c2 = model_color('lstm', list(reversed(names)))
    assert c1 == c2


def test_model_color_differs_across_models():
    names = ['lstm', 'transformer']
    assert model_color('lstm', names) != model_color('transformer', names)


# ── pick_primary_metric ──────────────────────────────────────────────────────

def test_pick_primary_metric_prefers_declared_when_present():
    models = {'lstm': {'custom_score': 0.9, 'rrmse_w': 0.1}}
    assert pick_primary_metric(models, declared='custom_score') == 'custom_score'


def test_pick_primary_metric_ignores_declared_when_absent_from_models():
    models = {'lstm': {'rrmse_w': 0.1}}
    assert pick_primary_metric(models, declared='not_a_real_metric') == 'rrmse_w'


def test_pick_primary_metric_falls_back_to_first_scalar_field():
    models = {'lstm': {'oddball_metric': 5}}
    assert pick_primary_metric(models) == 'oddball_metric'


def test_pick_primary_metric_returns_none_when_no_scalar_fields_exist():
    models = {'lstm': {'nested': {'a': 1}}}
    assert pick_primary_metric(models) is None


def test_pick_primary_metric_falls_back_to_subset_rrmse_w_before_plain_rrmse_w():
    """Regression: notebooks declare primary_metric='subset_rrmse_w' (config-
    driven), but CrossVal's own field is 'subset_rrmse_w_mean' -- a literal
    mismatch, so the declared check misses and this fallback needs to win
    over the plain 'rrmse_w' candidate that used to be tried first."""
    models = {'lstm': {'subset_rrmse_w': 0.05, 'rrmse_w': 0.1}}
    assert pick_primary_metric(models, declared='subset_rrmse_w_mean') == 'subset_rrmse_w'


# ── describe_primary_metric ──────────────────────────────────────────────────

def test_describe_primary_metric_empty_when_no_metric():
    assert describe_primary_metric(None) == ''


def test_describe_primary_metric_full_output_set():
    caption = describe_primary_metric('rrmse_w', 'single_split')
    assert caption == 'rrmse_w — the full output set, held-out test split'


def test_describe_primary_metric_subset_with_known_size():
    caption = describe_primary_metric('subset_rrmse_w', 'cross_validation', subset=list(range(20)))
    assert caption == 'subset_rrmse_w — the 20-output reported subset, averaged across CV folds'


def test_describe_primary_metric_subset_without_size_falls_back_to_generic_wording():
    caption = describe_primary_metric('subset_rrmse_w', 'single_split')
    assert caption == 'subset_rrmse_w — the reported subset, held-out test split'


def test_describe_primary_metric_unknown_eval_type_omits_basis_clause():
    caption = describe_primary_metric('rrmse_w', 'cross_stratum_transfer')
    assert caption == 'rrmse_w — the full output set'


# ── bar_chart_svg ─────────────────────────────────────────────────────────────

def test_bar_chart_svg_empty_when_metric_not_numeric_anywhere():
    assert bar_chart_svg({'lstm': {'rrmse_w': None}}, 'rrmse_w') == ''


def test_bar_chart_svg_contains_one_bar_per_model():
    svg = bar_chart_svg({'lstm': {'rrmse_w': 0.1}, 'cnn_lstm': {'rrmse_w': 0.2}}, 'rrmse_w')
    assert svg.count('<rect') == 2
    assert '<svg' in svg


def test_bar_chart_svg_draws_reference_line_when_nonzero():
    svg = bar_chart_svg({'lstm': {'dice_mean': 0.5}}, 'dice_mean', reference_value=1.0)
    assert '<line' in svg


def test_bar_chart_svg_skips_reference_line_at_zero():
    """A 0 = ideal reference already coincides with the bars' own left edge --
    drawing a line on top of it would show nothing new, so it's suppressed."""
    svg = bar_chart_svg({'lstm': {'rrmse_w': 0.1}}, 'rrmse_w', reference_value=0.0)
    assert '<line' not in svg


def test_bar_chart_svg_reference_value_expands_scale():
    """A reference far outside the data range (e.g. dice=1.0 vs actual ~0.1)
    must rescale so the bar stays within the plot area, not overflow it."""
    import re
    svg = bar_chart_svg({'lstm': {'dice_mean': 0.1}}, 'dice_mean', reference_value=1.0,
                        width=460, label_width=110)
    bar_width = float(re.search(r'to="([\d.]+)"', svg).group(1))
    plot_width = 460 - 110 - 60
    assert 0 < bar_width <= plot_width


# ── infer_reference ──────────────────────────────────────────────────────────

def test_infer_reference_dice_r2_and_auc_are_ceilinged_at_one():
    assert infer_reference('dice_mean') == {'value': 1.0, 'label': 'ideal'}
    assert infer_reference('r2_mean') == {'value': 1.0, 'label': 'ideal'}
    assert infer_reference('auc_mean') == {'value': 1.0, 'label': 'ideal'}
    assert infer_reference('subset_auc_mean') == {'value': 1.0, 'label': 'ideal'}


def test_infer_reference_rrmse_and_mse_are_floored_at_zero():
    assert infer_reference('rrmse_w')['value'] == 0.0
    assert infer_reference('test_mse')['value'] == 0.0


def test_infer_reference_returns_none_for_unrecognized_metric():
    assert infer_reference('some_future_metric_nobody_has_seen_yet') is None


# ── build_per_output_charts ──────────────────────────────────────────────────

def _big_flat_dict(prefix, n=10):
    return {f'{prefix}{i}': float(i) / 10 for i in range(n)}


def test_build_per_output_charts_detects_large_flat_dicts_only():
    models = {
        'lstm': {'rrmse_w': 0.1, 'per_output_rrmse': _big_flat_dict('m')},
        'cnn_lstm': {'small_dict': {'a': 1}, 'per_output_rrmse': _big_flat_dict('m')},
    }
    charts = build_per_output_charts(models)
    field_names = {c['field_name'] for c in charts}
    assert field_names == {'per_output_rrmse'}   # small_dict has too few keys


def test_build_per_output_charts_sorts_worst_first():
    models = {'lstm': {'per_output_rrmse': {**_big_flat_dict('m'), 'zz_worst': 99.0}}}
    charts = build_per_output_charts(models)
    assert charts[0]['rows'][0]['key'] == 'zz_worst'


def test_build_per_output_charts_missing_key_in_one_model_gets_zero_pct_bar():
    models = {
        'a': {'per_output_rrmse': _big_flat_dict('m')},
        'b': {'per_output_rrmse': {k: v for k, v in _big_flat_dict('m').items() if k != 'm0'}},
    }
    charts = build_per_output_charts(models)
    row = next(r for r in charts[0]['rows'] if r['key'] == 'm0')
    b_bar = next(b for b in row['bars'] if b['model'] == 'b')
    assert b_bar['value'] is None
    assert b_bar['pct'] == 0.0


def test_build_per_output_charts_pathological_outlier_is_clamped_not_scaled_to():
    """Regression case for real per-output R2 data: one output at -17 while the
    rest sit in [-1, 1]. The outlier's own bar clamps at 100%; it must not
    become the scale that crushes every normal-range bar to near-zero."""
    normal = {f'm{i}': (i - 5) / 10 for i in range(20)}   # spread roughly -0.5..1.0
    models = {'lstm': {'per_output_r2': {**normal, 'zz_outlier': -17.2}}}
    charts = build_per_output_charts(models)
    rows = {r['key']: r for r in charts[0]['rows']}

    outlier_pct = rows['zz_outlier']['bars'][0]['pct']
    assert outlier_pct == 100.0

    typical_pcts = [rows[k]['bars'][0]['pct'] for k in normal if abs(normal[k]) > 0.3]
    assert all(p > 15.0 for p in typical_pcts), typical_pcts


def test_build_per_output_charts_outlier_value_still_shown_exactly():
    """Clamped bar width shouldn't hide the real number -- it's still exact."""
    models = {'lstm': {'per_output_r2': {**_big_flat_dict('m'), 'zz_outlier': -17.2}}}
    charts = build_per_output_charts(models)
    row = next(r for r in charts[0]['rows'] if r['key'] == 'zz_outlier')
    assert row['bars'][0]['value'] == -17.2


def test_build_per_output_charts_tags_subset_membership():
    models = {'lstm': {'per_output_rrmse': _big_flat_dict('m')}}
    charts = build_per_output_charts(models, subset=['m0', 'm1'])
    rows_by_key = {r['key']: r for r in charts[0]['rows']}
    assert rows_by_key['m0']['in_subset'] is True
    assert rows_by_key['m1']['in_subset'] is True
    assert rows_by_key['m2']['in_subset'] is False


def test_build_per_output_charts_in_subset_false_when_no_subset_given():
    """No `subset` arg at all (the default) -- every row should read False,
    not None/missing, so templates can safely check `row.in_subset`."""
    models = {'lstm': {'per_output_rrmse': _big_flat_dict('m')}}
    charts = build_per_output_charts(models)
    assert all(r['in_subset'] is False for r in charts[0]['rows'])


# ── waveform_svg ─────────────────────────────────────────────────────────────

def test_waveform_svg_empty_when_no_numeric_values():
    assert waveform_svg({'Ground truth': [None, None]}) == ''


def test_waveform_svg_one_polyline_per_series():
    svg = waveform_svg({'Ground truth': [0.1, 0.2, 0.3], 'lstm': [0.15, 0.25, 0.28]})
    assert svg.count('<polyline') == 2
    assert '<svg' in svg


def test_waveform_svg_ground_truth_uses_currentcolor_not_model_palette():
    svg = waveform_svg({'Ground truth': [0.1, 0.2], 'lstm': [0.1, 0.2]})
    gt_line = svg.split('<polyline')[1]
    assert 'stroke="currentColor"' in gt_line


def test_waveform_svg_handles_series_of_different_lengths():
    """A raw uploaded trial and a 100-point normalized-stance sample won't be
    the same length -- each series should still plot over its own full width."""
    svg = waveform_svg({'a': [0.0, 1.0], 'b': [0.0, 0.5, 1.0, 0.5, 0.0]})
    assert svg.count('<polyline') == 2


def test_waveform_svg_highlight_regions_render_as_rects():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]}, highlight_regions=[(0.1, 0.4), (0.6, 0.9)])
    assert svg.count('<rect') == 2


def test_waveform_svg_fill_between_draws_polygon_before_lines():
    """The shaded region must render UNDER the polylines (earlier in the SVG
    markup = painted first), so the curves stay visible on top of the fill."""
    svg = waveform_svg(
        {'Ground truth': [0.1, 0.5, 0.9], 'lstm': [0.2, 0.4, 0.7]},
        fill_between=('Ground truth', 'lstm'),
    )
    assert svg.count('<polygon') == 1
    assert svg.index('<polygon') < svg.index('<polyline')


def test_waveform_svg_fill_between_absent_series_produces_no_polygon():
    svg = waveform_svg({'Ground truth': [0.1, 0.5, 0.9]},
                       fill_between=('Ground truth', 'does_not_exist'))
    assert '<polygon' not in svg


def test_waveform_svg_fill_between_mismatched_length_produces_no_polygon():
    svg = waveform_svg(
        {'Ground truth': [0.1, 0.5, 0.9], 'lstm': [0.2, 0.4]},
        fill_between=('Ground truth', 'lstm'),
    )
    assert '<polygon' not in svg


def test_waveform_svg_fill_between_uses_the_second_series_model_color():
    svg = waveform_svg(
        {'Ground truth': [0.1, 0.5], 'lstm': [0.2, 0.4], 'transformer': [0.3, 0.6]},
        fill_between=('Ground truth', 'transformer'),
    )
    polygon = svg[svg.index('<polygon'):svg.index('/>', svg.index('<polygon'))]
    assert f'fill="{model_color("transformer", ["lstm", "transformer"])}"' in polygon


def test_waveform_svg_default_axis_labels_present():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]})
    assert '% stance' in svg
    assert 'Force (N/kg)' in svg
    assert svg.count('class="axis-label"') == 2


def test_waveform_svg_axis_labels_can_be_omitted():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]}, x_label=None, y_label=None)
    assert 'axis-label' not in svg
    assert 'axis-tick' not in svg
    assert 'axis-line' not in svg


def test_waveform_svg_custom_axis_labels():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]}, x_label='time (s)', y_label='raw GRF (N)')
    assert 'time (s)' in svg
    assert 'raw GRF (N)' in svg
    assert '% stance' not in svg


def test_waveform_svg_draws_axis_lines():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]})
    assert svg.count('class="axis-line"') == 2


def test_waveform_svg_ytick_values_are_series_extremes():
    svg = waveform_svg({'a': [0.1, 0.5, 0.9]})
    assert svg.count('class="axis-tick"') == 4
    assert '>0.90<' in svg
    assert '>0.10<' in svg


def test_waveform_svg_xtick_values_default_to_0_100():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]})
    assert '>0<' in svg
    assert '>100<' in svg


def test_waveform_svg_xtick_values_use_custom_x_range():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]}, x_range=(0, 1))
    assert '>0<' in svg
    assert '>1<' in svg
    assert '>100<' not in svg


def test_waveform_svg_no_ticks_or_lines_when_axis_omitted_individually():
    svg = waveform_svg({'a': [0.1, 0.2, 0.3]}, y_label=None)
    assert svg.count('class="axis-line"') == 1
    assert svg.count('class="axis-tick"') == 2


# ── slope_chart_svg ──────────────────────────────────────────────────────────

def test_slope_chart_svg_empty_when_no_pairs():
    assert slope_chart_svg([]) == ''


def test_slope_chart_svg_one_line_group_per_row():
    svg = slope_chart_svg([
        {'subject': 'Subject1', 'baseline': 0.5, 'retention': 0.4},
        {'subject': 'Subject2', 'baseline': 0.3, 'retention': 0.35},
    ])
    assert svg.count('class="slope-row"') == 2
    assert svg.count('<line') == 2
    assert svg.count('<circle') == 4


def test_slope_chart_svg_title_carries_label_and_values():
    svg = slope_chart_svg([{'subject': 'Subject1', 'baseline': 0.5, 'retention': 0.4}])
    assert '<title>Subject1: 0.500 → 0.400</title>' in svg


def test_slope_chart_svg_custom_keys_and_labels():
    svg = slope_chart_svg(
        [{'name': 'a', 'x': 1.0, 'y': 2.0}],
        left_key='x', right_key='y', left_label='pre', right_label='post', label_key='name',
    )
    assert 'pre' in svg and 'post' in svg
    assert '<title>a: 1.000 → 2.000</title>' in svg


def test_slope_chart_svg_per_row_style_overrides():
    svg = slope_chart_svg([
        {'subject': 'a', 'baseline': 0.1, 'baseline_2': None, 'retention': 0.2,
         'color': '#AB202D', 'opacity': 0.9, 'stroke_width': 2.5},
        {'subject': 'b', 'baseline': 0.3, 'retention': 0.1, 'dash': '4 3', 'flagged': True},
    ])
    assert 'stroke="#AB202D"' in svg
    assert 'stroke-opacity="0.9"' in svg
    assert 'stroke-width="2.5"' in svg
    assert 'stroke-dasharray="4 3"' in svg
    assert 'class="slope-row slope-flagged"' in svg
    assert svg.count('class="slope-row"') == 1   # only the non-flagged row


# ── build_ratio_prediction_panels ────────────────────────────────────────────

def _prediction_data(subjects=('Subject1', 'Subject2'), model_names=('lstm',)):
    ground_truth = {s: {'baseline': 0.5, 'retention': 0.4} for s in subjects}
    predictions = {
        m: {
            'base_trained': {s: {'baseline': 0.5, 'retention': 0.4} for s in subjects},
            'ret_trained': {s: {'baseline': 0.5, 'retention': 0.4} for s in subjects},
        }
        for m in model_names
    }
    return {'subjects': list(subjects), 'ground_truth': ground_truth, 'predictions': predictions}


def _ratio_card(model='lstm', loa=(-0.1, 0.1), spearman_r=0.9):
    return {
        'model': model, 'color': '#AB202D',
        'conditions': [
            {'label': 'base_trained', 'loa_lower': loa[0], 'loa_upper': loa[1],
             'spearman_r': spearman_r},
            {'label': 'ret_trained', 'loa_lower': loa[0], 'loa_upper': loa[1],
             'spearman_r': spearman_r},
        ],
    }


def test_build_ratio_prediction_panels_one_panel_per_subject():
    result = build_ratio_prediction_panels(_prediction_data(), [_ratio_card()], ['lstm'])
    panels = result['panels']
    assert [p['subject'] for p in panels] == ['Subject1', 'Subject2']
    assert all(p['svg'] for p in panels)


def test_build_ratio_prediction_panels_picks_best_model_by_spearman_r():
    data = _prediction_data(model_names=('lstm', 'cnn_lstm'))
    cards = [_ratio_card('lstm', spearman_r=0.5), _ratio_card('cnn_lstm', spearman_r=0.9)]
    result = build_ratio_prediction_panels(data, cards, ['lstm', 'cnn_lstm'])
    assert result['best_models']['base_trained']['model'] == 'cnn_lstm'
    assert result['best_models']['ret_trained']['model'] == 'cnn_lstm'
    # Only the winning model's line + ground truth -- not every architecture.
    assert result['panels'][0]['svg'].count('class="slope-row') == 3


def test_build_ratio_prediction_panels_no_best_model_without_spearman_r():
    card = {'model': 'lstm', 'color': '#AB202D',
           'conditions': [{'label': 'base_trained', 'loa_lower': -0.1, 'loa_upper': 0.1}]}
    result = build_ratio_prediction_panels(_prediction_data(), [card], ['lstm'])
    assert result['best_models']['base_trained'] is None
    assert result['best_models']['ret_trained'] is None


def test_build_ratio_prediction_panels_flags_out_of_loa_subject():
    """true shift = 0.4-0.5 = -0.1; predicted shift is identical (-0.1) so
    diff=0 -- comfortably inside a [-0.1, 0.1] LOA band -- must NOT flag."""
    result = build_ratio_prediction_panels(
        _prediction_data(), [_ratio_card(loa=(-0.05, 0.05))], ['lstm'],
    )
    assert all(not p['any_flagged'] for p in result['panels'])


def test_build_ratio_prediction_panels_flags_when_diff_exceeds_loa():
    data = _prediction_data()
    data['predictions']['lstm']['base_trained']['Subject1']['retention'] = 0.9   # big diff now
    result = build_ratio_prediction_panels(data, [_ratio_card(loa=(-0.05, 0.05))], ['lstm'])
    by_subject = {p['subject']: p for p in result['panels']}
    assert by_subject['Subject1']['any_flagged'] is True
    assert by_subject['Subject2']['any_flagged'] is False


def test_build_ratio_prediction_panels_skips_subject_missing_ground_truth():
    data = _prediction_data()
    del data['ground_truth']['Subject1']
    result = build_ratio_prediction_panels(data, [_ratio_card()], ['lstm'])
    assert [p['subject'] for p in result['panels']] == ['Subject2']


def test_build_ratio_prediction_panels_empty_predictions_returns_empty_list():
    result = build_ratio_prediction_panels(
        {'subjects': [], 'ground_truth': {}, 'predictions': {}}, [], [],
    )
    assert result == {
        'panels': [],
        'best_models': {'base_trained': None, 'ret_trained': None},
    }


# ── build_transfer_comparison ────────────────────────────────────────────────
# Cross-stratum transfer results (CrossVal.ipynb's RUN_TRANSFER export) are
# keyed by direction (base_base/base_ret/ret_ret/ret_base) rather than the
# flat {metric: value} shape build_models_table assumes, plus a
# ratio_recovery block of paired correlation/bias/spread-ratio stats.

def _direction(rrmse_mean=0.10, rrmse_std=0.01):
    return {
        'direction': 'x->y', 'rrmse_w_mean': rrmse_mean, 'rrmse_w_std': rrmse_std,
        'clinical': {'knee_fy': {'mae_mean': 1.0, 'mae_over_threshold_mean': 0.2,
                                 'threshold': 5.0}},
    }


def _ratio_condition(spearman_p=0.001, spread_ratio=0.4):
    return {
        'n': 10, 'pearson_r': 0.8, 'pearson_p': 0.01,
        'spearman_r': 0.9, 'spearman_p': spearman_p,
        'bias': 0.02, 'loa_lower': -0.1, 'loa_upper': 0.15,
        'std_pred': 0.03, 'std_gt': 0.08, 'spread_ratio': spread_ratio,
    }


def _transfer_model(ratio_recovery=True, ret_trained=True, spearman_p=0.001,
                    spread_ratio=0.4, pooled=False):
    entry = {
        'base_base': _direction(), 'base_ret': _direction(0.12, 0.02),
        'ret_ret': _direction(0.11, 0.01), 'ret_base': _direction(),
    }
    if pooled:
        entry['pooled_base'] = _direction(0.13, 0.02)
        entry['pooled_ret'] = _direction(0.14, 0.02)
    if ratio_recovery:
        entry['ratio_recovery'] = {
            'base_trained': _ratio_condition(spearman_p, spread_ratio),
            'ret_trained': _ratio_condition(0.9, 0.5) if ret_trained else None,
            'transfer_cost': {
                'ratio_transfer_cost_bias': 0.01, 'rrmse_w_base2ret': 0.12,
                'rrmse_w_ret2ret': 0.11, 'rrmse_w_transfer_cost': 0.01,
            },
        }
        if pooled:
            entry['ratio_recovery']['pooled_trained'] = _ratio_condition(0.02, 0.6)
    return entry


def test_build_transfer_comparison_discovers_directions_in_preferred_order():
    ctx = build_transfer_comparison({'lstm': _transfer_model()})
    assert ctx['directions'] == ['base_base', 'base_ret', 'ret_ret', 'ret_base']


def test_build_transfer_comparison_rrmse_table_carries_full_direction_dict():
    ctx = build_transfer_comparison({'lstm': _transfer_model()})
    row = next(r for r in ctx['rrmse_table'] if r['direction'] == 'base_ret')
    assert row['by_model']['lstm']['rrmse_w_mean'] == 0.12
    assert 'clinical' in row['by_model']['lstm']   # full dict, not just rrmse fields


def test_build_transfer_comparison_skips_model_with_no_ratio_recovery():
    ctx = build_transfer_comparison({
        'lstm': _transfer_model(ratio_recovery=True),
        'cnn_lstm': _transfer_model(ratio_recovery=False),
    })
    cards_by_model = {c['model'] for c in ctx['ratio_cards']}
    assert cards_by_model == {'lstm'}


def test_build_transfer_comparison_skips_condition_that_is_none():
    """ratio_recovery's conditions are built via .get() in the exporter --
    a model can legitimately have base_trained but not ret_trained."""
    ctx = build_transfer_comparison({'lstm': _transfer_model(ret_trained=False)})
    card = ctx['ratio_cards'][0]
    labels = {c['label'] for c in card['conditions']}
    assert labels == {'base_trained'}


def test_build_transfer_comparison_significant_flag_boundary():
    below = build_transfer_comparison({'m': _transfer_model(spearman_p=0.0499)})
    at = build_transfer_comparison({'m': _transfer_model(spearman_p=0.05)})
    above = build_transfer_comparison({'m': _transfer_model(spearman_p=0.06)})
    assert below['ratio_cards'][0]['conditions'][0]['significant'] is True
    assert at['ratio_cards'][0]['conditions'][0]['significant'] is False
    assert above['ratio_cards'][0]['conditions'][0]['significant'] is False


def test_build_transfer_comparison_spread_ratio_pct_scaling():
    """Fixed 0..2 scale, 1.0 (full fidelity) sits at 50% width; clamps at 100%."""
    ctx = build_transfer_comparison({'m': _transfer_model(spread_ratio=1.0)})
    assert ctx['ratio_cards'][0]['conditions'][0]['spread_ratio_pct'] == 50.0

    ctx = build_transfer_comparison({'m': _transfer_model(spread_ratio=5.0)})
    assert ctx['ratio_cards'][0]['conditions'][0]['spread_ratio_pct'] == 100.0


def test_build_transfer_comparison_model_color_matches_model_color_fn():
    ctx = build_transfer_comparison({'lstm': _transfer_model(), 'cnn_lstm': _transfer_model()})
    for card in ctx['ratio_cards']:
        assert card['color'] == model_color(card['model'], ['lstm', 'cnn_lstm'])


def test_build_transfer_comparison_discovers_pooled_directions():
    """pooled_base/pooled_ret (model trained on baseline+retention combined,
    tested on each condition separately) are a new addition alongside the
    original four directions -- must show up as extra rrmse_table rows."""
    ctx = build_transfer_comparison({'lstm': _transfer_model(pooled=True)})
    assert ctx['directions'] == [
        'base_base', 'base_ret', 'ret_ret', 'ret_base', 'pooled_base', 'pooled_ret',
    ]


def test_build_transfer_comparison_ratio_cards_include_pooled_trained_condition():
    """ratio_recovery now has three sub-keys (base_trained/ret_trained/
    pooled_trained), not two -- pooled_trained must appear as a third
    per-model condition, not get silently dropped."""
    ctx = build_transfer_comparison({'lstm': _transfer_model(pooled=True)})
    labels = [c['label'] for c in ctx['ratio_cards'][0]['conditions']]
    assert labels == ['base_trained', 'ret_trained', 'pooled_trained']
