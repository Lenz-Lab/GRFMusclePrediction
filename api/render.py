"""Formatting/table-building helpers for the HTML browsing views.

Kept separate from data_access.py: that module's job is reading arbitrary
metrics_doc shapes without assumptions; this module's job is turning that
generic data into something a human can scan, still without assuming which
fields exist (see the schema-light rationale in data_access.py's docstring).
"""
from typing import Any, Optional


def fmt_value(value: Any) -> str:
    if value is None:
        return '—'   # em dash
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


# ── Units ────────────────────────────────────────────────────────────────────
# Every reconstruction-accuracy metric except MAE is a normalized ratio
# (RRMSE/R2/Dice/AUC all divide out the signal's own scale), so it's
# dimensionless -- no unit to show. MAE is the one metric still in the
# signal's native force units (N/kg, since inputs/outputs are mass-normalized
# -- see grf_pipeline_utils/data_utils.py's normalize_by_mass_in_order), and
# so is a clinical threshold compared directly against an MAE value.

def _in_mae_units(key: str) -> bool:
    k = key.lower()
    if 'over_threshold' in k or 'p_value' in k or 'ratio' in k:
        return False   # dimensionless -- a ratio derived FROM an MAE, not an MAE
    return 'mae' in k or k == 'threshold'


def fmt_with_unit(value: Any, key: Optional[str] = None) -> str:
    """Same as fmt_value, but appends 'N/kg' when `key` names a field that's
    actually in force units (see _in_mae_units) -- used wherever a metric's
    field name is available alongside its value."""
    text = fmt_value(value)
    if key and isinstance(value, (int, float)) and not isinstance(value, bool) \
            and _in_mae_units(key):
        return f'{text} N/kg'
    return text


def scalar_list_or_none(value: list) -> str | None:
    """A comma-joined string if every item is a plain scalar, else None (caller
    should fall back to rendering it as a nested structure)."""
    if not all(not isinstance(v, (dict, list)) for v in value):
        return None
    return ', '.join(fmt_value(v) for v in value)


def build_models_table(models: dict[str, dict],
                       exclude_keys: frozenset = frozenset()) -> dict[str, Any]:
    """Splits each model's metrics into scalar fields (one side-by-side
    comparison table, columns = models) vs. nested dict/list fields (rendered
    separately per model, since e.g. 'by_group' or 'per_fold_rrmse_w' aren't
    guaranteed to have the same shape across every eval_type).

    `exclude_keys` drops fields already handled elsewhere (e.g. large
    per-output metric maps get their own interactive chart -- see
    build_per_output_charts -- so they shouldn't also show up here)."""
    model_names = sorted(models.keys())

    scalar_keys: list[str] = []
    seen = set()
    for name in model_names:
        for k, v in models[name].items():
            if k in exclude_keys:
                continue
            if not isinstance(v, (dict, list)) and k not in seen:
                scalar_keys.append(k)
                seen.add(k)

    # NOTE: key is 'by_model', not 'values' -- Jinja2's attribute/item fallback
    # would otherwise resolve `row.values` to dict.values() (a bound method),
    # not this dict's 'values' item, silently breaking the template.
    rows = [
        {'metric': k, 'by_model': {name: models[name].get(k) for name in model_names}}
        for k in scalar_keys
    ]
    nested = {
        name: {k: v for k, v in models[name].items()
               if isinstance(v, (dict, list)) and k not in exclude_keys}
        for name in model_names
    }
    return {'model_names': model_names, 'rows': rows, 'nested': nested}


# ── Reference values ─────────────────────────────────────────────────────────
# Only the literal mathematical bound of a well-known metric is surfaced (Dice,
# R², and AUC/integral-agreement are all ceilinged at 1.0; RRMSE/MSE are
# floored at 0.0) -- never a generic "R² > 0.7 is good" clinical-style
# threshold, since there's nothing in this project backing those. (Previously
# also surfaced a flexor_ratio_ground_truth reference, retired along with the
# flexor-ratio metric itself -- see notebooks/Compare_Models.ipynb, now AUC-
# based instead.)

def infer_reference(metric_name: str, extra: dict[str, Any] = None) -> Optional[dict[str, Any]]:
    name = metric_name.lower()

    if 'dice' in name or 'r2' in name or 'auc' in name:
        return {'value': 1.0, 'label': 'ideal'}
    if 'rrmse' in name or 'rmse' in name or name.endswith('mse'):
        return {'value': 0.0, 'label': 'ideal (0 = perfect)'}
    return None


# ── Charts ───────────────────────────────────────────────────────────────────
# Plain inline SVG, built server-side -- no JS charting library, no CDN, so the
# page still works with charts even if you're running this fully offline.

MODEL_COLOR_PALETTE = ['#AB202D', '#0D9488', '#EA580C', '#DB2777', '#65A30D', '#7C3AED']
# maroon, teal, orange, pink, green, violet -- maroon matches base.html's --accent

_PRIMARY_METRIC_CANDIDATES = ('subset_rrmse_w', 'rrmse_w', 'rrmse_w_mean', 'test_mse',
                              'dice_mean', 'r2_mean')


def model_color(model_name: str, model_names: list[str]) -> str:
    """Same model -> same color everywhere, so a model's identity is visually
    consistent across the list page, detail page, and every chart."""
    ordered = sorted(model_names)
    idx = ordered.index(model_name) if model_name in ordered else 0
    return MODEL_COLOR_PALETTE[idx % len(MODEL_COLOR_PALETTE)]


def pick_primary_metric(models: dict[str, dict], declared: Any = None) -> Optional[str]:
    """Which scalar metric headlines the chart. Prefers a metrics_doc's own
    self-declared 'primary_metric' (CrossVal.ipynb already sets this), then a
    short list of common names, then whatever scalar field appears first --
    never assumes a fixed schema, since the metric set is still evolving."""
    if isinstance(declared, str) and any(declared in m for m in models.values()):
        return declared
    for cand in _PRIMARY_METRIC_CANDIDATES:
        if any(cand in m for m in models.values()):
            return cand
    for m in models.values():
        for k, v in m.items():
            if not isinstance(v, (dict, list)):
                return k
    return None


_EVAL_TYPE_BASIS = {
    'single_split': 'held-out test split',
    'cross_validation': 'averaged across CV folds',
}


def describe_primary_metric(metric: Optional[str], eval_type: Optional[str] = None,
                            subset: Optional[list] = None) -> str:
    """A one-line provenance caption for the headline metric shown on a
    results-list card -- e.g. 'subset_rrmse_w -- the 20-output reported
    subset, averaged across CV folds'. Exists because the card alone (a name
    + a bar chart) doesn't say whether that number covers all outputs or
    just the reported subset, or what it's averaged over -- easy to
    misread as "the" RRMSE when it's one specific slice of it."""
    if not metric:
        return ''
    is_subset = 'subset' in metric.lower()
    if is_subset and subset:
        scope = f'the {len(subset)}-output reported subset'
    elif is_subset:
        scope = 'the reported subset'
    else:
        scope = 'the full output set'
    basis = _EVAL_TYPE_BASIS.get(eval_type)
    return f'{metric} — {scope}' + (f', {basis}' if basis else '')


def bar_chart_svg(models: dict[str, dict], metric: str, *, width: int = 460,
                  bar_height: int = 24, gap: int = 8, label_width: int = 110,
                  reference_value: Optional[float] = None) -> str:
    """Self-contained horizontal bar chart for one scalar metric across models.
    `reference_value`, if given, is drawn as a dashed vertical line (e.g. the
    mathematical ideal for a bounded metric -- see infer_reference)."""
    model_names = sorted(models.keys())
    pairs = [(name, models[name].get(metric)) for name in model_names]
    numeric = [v for _, v in pairs if isinstance(v, (int, float))]
    if not numeric:
        return ''

    scale_values = list(numeric)
    if isinstance(reference_value, (int, float)):
        scale_values.append(reference_value)
    max_abs = max(abs(v) for v in scale_values) or 1.0
    plot_width = max(60, width - label_width - 60)
    height = len(pairs) * (bar_height + gap)
    rows = []
    for i, (name, value) in enumerate(pairs):
        y = i * (bar_height + gap)
        color = model_color(name, model_names)
        if isinstance(value, (int, float)):
            bw = max(2.0, abs(value) / max_abs * plot_width)
            text = fmt_with_unit(value, metric)
        else:
            bw, text = 0.0, '—'
        rows.append(
            f'<text x="0" y="{y + bar_height * 0.68:.1f}" class="bar-label">{name}</text>'
            f'<rect x="{label_width}" y="{y}" width="0" height="{bar_height}" rx="5" '
            f'fill="{color}"><animate attributeName="width" from="0" to="{bw:.1f}" '
            f'dur="0.6s" fill="freeze"/></rect>'
            f'<text x="{label_width + bw + 8:.1f}" y="{y + bar_height * 0.68:.1f}" '
            f'class="bar-value">{text}</text>'
        )
    # A reference at 0 coincides with the bars' own left edge (they're already
    # anchored there) -- drawing a line on top of it would show nothing new.
    ref_svg = ''
    if isinstance(reference_value, (int, float)) and reference_value != 0:
        rx = label_width + abs(reference_value) / max_abs * plot_width
        ref_svg = (f'<line x1="{rx:.1f}" y1="0" x2="{rx:.1f}" y2="{height}" '
                  f'stroke="currentColor" stroke-dasharray="4 3" stroke-opacity="0.5"/>')
    svg_width = label_width + plot_width + 80
    return (
        f'<svg viewBox="0 0 {svg_width} {height}" width="{svg_width}" height="{height}" '
        f'class="barchart" role="img" aria-label="{metric} by model">'
        + ref_svg + ''.join(rows) + '</svg>'
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


def _robust_scale(values: list[float], pct: float = 95.0) -> float:
    """A percentile-based normalization denominator rather than the true max.
    Metrics like per-output R2 are unbounded below, and a single pathological
    output (seen in practice: R2 as low as -17 on one muscle while everything
    else sits between -1 and 1) would otherwise crush every other bar to
    near-invisible width. Outlier bars still just clip at 100% width -- their
    real value is always shown as text, so nothing is hidden, just not
    allowed to dictate the whole chart's scale."""
    if not values:
        return 1.0
    p = _percentile(values, pct)
    return p if p > 0 else (max(values) or 1.0)


# ── Per-output charts ────────────────────────────────────────────────────────
# Large flat scalar maps (e.g. per_output_rrmse: 40+ muscle/joint keys) are
# unreadable as a scrolled table. Render them as compact, filterable,
# worst-first bar lists instead -- one combined chart per field, grouped by
# model, rather than a separate 40-row dump per model.

def build_per_output_charts(models: dict[str, dict], min_items: int = 8,
                            subset: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """`subset`, if given (a result's own reported-subset list), tags each row
    with `in_subset` so the template can flag which outputs are actually
    headlined -- purely a label, doesn't change the worst-first sort order."""
    subset_set = set(subset) if subset else None

    fields: dict[str, dict[str, dict]] = {}
    for model_name, metrics in models.items():
        for key, value in metrics.items():
            if (isinstance(value, dict) and len(value) >= min_items
                    and all(isinstance(v, (int, float)) for v in value.values())):
                fields.setdefault(key, {})[model_name] = value

    charts = []
    for field_name, per_model in fields.items():
        model_names = sorted(per_model.keys())
        output_keys = sorted({k for d in per_model.values() for k in d})
        all_abs = [abs(v) for d in per_model.values() for v in d.values()]
        max_abs = _robust_scale(all_abs)

        rows = []
        for key in output_keys:
            by_model = {name: per_model[name].get(key) for name in model_names}
            worst = max((abs(v) for v in by_model.values() if isinstance(v, (int, float))),
                       default=0.0)
            rows.append({
                'key': key,
                'worst': worst,
                'in_subset': subset_set is not None and key in subset_set,
                'bars': [
                    {
                        'model': name,
                        'value': by_model[name],
                        'pct': min(100.0, round(abs(by_model[name]) / max_abs * 100, 1))
                               if isinstance(by_model[name], (int, float)) else 0.0,
                        'color': model_color(name, model_names),
                    }
                    for name in model_names
                ],
            })
        rows.sort(key=lambda r: r['worst'], reverse=True)
        charts.append({
            'field_name': field_name,
            'model_names': [{'name': n, 'color': model_color(n, model_names)}
                            for n in model_names],
            'rows': rows,
        })
    return charts


# ── Waveform (line) charts ────────────────────────────────────────────────────
# Everything above is bar charts (one scalar per model/output). Live inference
# needs a curve over time instead -- one <polyline> per series on a shared 0..1
# x-axis, so it works equally for a 100-point normalized-stance sample and an
# arbitrary-length uploaded raw trial.

_GROUND_TRUTH_NAMES = frozenset({'ground truth', 'y_true', 'ground_truth'})


def waveform_svg(series: dict[str, list[float]], *, width: int = 640, height: int = 220,
                 highlight_regions: Optional[list[tuple[float, float]]] = None,
                 fill_between: Optional[tuple[str, str]] = None,
                 x_label: Optional[str] = '% stance',
                 y_label: Optional[str] = 'Force (N/kg)',
                 x_range: tuple[float, float] = (0, 100)) -> str:
    """`series`: {label: [values...]}, each plotted over its own length on a
    shared 0..1 x-axis. A label matching one of _GROUND_TRUTH_NAMES (case-
    insensitive) is drawn in black/bold; everything else cycles model_color().
    `highlight_regions`: [(x0, x1), ...] in the same 0..1 x-axis fraction,
    shaded behind the lines -- used for detected stance windows on a raw trace.
    `fill_between`: (name_a, name_b) -- if both are present in `series` with
    matching length, shades the region between their two curves (name_a's
    curve forward + name_b's reversed, closed into one polygon). A visual
    read on the same overlap-vs-discrepancy question calc_auc_per_output
    (eval_utils.py) answers numerically -- not a pixel-exact rendering of
    that trapezoidal ratio, just the same idea made visible.
    `x_label`/`y_label`: axis captions (defaults match every current caller --
    a gait-cycle sample is always force in N/kg over % stance); pass None to
    omit either one and reclaim its margin. Each labeled axis also draws a
    visible rule line plus its extreme values as tick text (0/100 for x, the
    series' own min/max for y) -- there was previously no line or number at
    all, just the caption, which read as "meaningless" on an unscaled chart.
    `x_range`: the (start, end) values shown at the x-axis ticks -- defaults
    to 0..100 to match the '% stance' convention every current caller uses.
    """
    all_vals = [v for vals in series.values() for v in vals if isinstance(v, (int, float))]
    if not all_vals:
        return ''

    y_min, y_max = min(all_vals), max(all_vals)
    y_range = (y_max - y_min) or 1.0
    pad_left = 70 if y_label else 10
    pad_bottom = 38 if x_label else 10
    pad_top, pad_right = 14, 16
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def _coords(vals):
        n = len(vals)
        out = []
        for i, v in enumerate(vals):
            x = pad_left + (i / max(1, n - 1)) * plot_w
            y = pad_top + (1 - (v - y_min) / y_range) * plot_h
            out.append((x, y))
        return out

    def _points_str(coords):
        return ' '.join(f'{x:.1f},{y:.1f}' for x, y in coords)

    model_names = [name for name in series if name.lower() not in _GROUND_TRUTH_NAMES]

    region_svg = ''.join(
        f'<rect x="{pad_left + x0 * plot_w:.1f}" y="{pad_top}" width="{(x1 - x0) * plot_w:.1f}" '
        f'height="{plot_h}" fill="currentColor" fill-opacity="0.08"/>'
        for x0, x1 in (highlight_regions or [])
    )

    fill_svg = ''
    if fill_between:
        name_a, name_b = fill_between
        vals_a, vals_b = series.get(name_a), series.get(name_b)
        if vals_a and vals_b and len(vals_a) == len(vals_b):
            reversed_b = _points_str(list(reversed(_coords(vals_b))))
            polygon_pts = f'{_points_str(_coords(vals_a))} {reversed_b}'
            fill_color = (model_color(name_b, model_names)
                         if name_b in model_names else 'currentColor')
            fill_svg = (f'<polygon points="{polygon_pts}" fill="{fill_color}" '
                       f'fill-opacity="0.15" stroke="none"/>')

    lines_svg = ''
    for name, vals in series.items():
        if name.lower() in _GROUND_TRUTH_NAMES:
            color, stroke_width = 'currentColor', 2.5
        else:
            color, stroke_width = model_color(name, model_names), 1.5
        lines_svg += (f'<polyline points="{_points_str(_coords(vals))}" fill="none" '
                     f'stroke="{color}" stroke-width="{stroke_width}"/>')

    axis_svg = ''
    if x_label:
        axis_y = pad_top + plot_h
        x0, x1 = x_range
        axis_svg += (f'<line x1="{pad_left:.1f}" y1="{axis_y:.1f}" x2="{pad_left + plot_w:.1f}" '
                    f'y2="{axis_y:.1f}" class="axis-line"/>'
                    f'<text x="{pad_left:.1f}" y="{axis_y + 13:.1f}" text-anchor="start" '
                    f'class="axis-tick">{x0:g}</text>'
                    f'<text x="{pad_left + plot_w:.1f}" y="{axis_y + 13:.1f}" text-anchor="end" '
                    f'class="axis-tick">{x1:g}</text>'
                    f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 4}" '
                    f'text-anchor="middle" class="axis-label">{x_label}</text>')
    if y_label:
        cy = pad_top + plot_h / 2
        axis_svg += (f'<line x1="{pad_left:.1f}" y1="{pad_top:.1f}" x2="{pad_left:.1f}" '
                    f'y2="{pad_top + plot_h:.1f}" class="axis-line"/>'
                    f'<text x="{pad_left - 6:.1f}" y="{pad_top + 4:.1f}" text-anchor="end" '
                    f'class="axis-tick">{y_max:.2f}</text>'
                    f'<text x="{pad_left - 6:.1f}" y="{pad_top + plot_h:.1f}" text-anchor="end" '
                    f'class="axis-tick">{y_min:.2f}</text>'
                    f'<text x="12" y="{cy:.1f}" text-anchor="middle" class="axis-label" '
                    f'transform="rotate(-90 12 {cy:.1f})">{y_label}</text>')

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'class="waveform" role="img" aria-label="waveform chart">'
        + region_svg + fill_svg + lines_svg + axis_svg + '</svg>'
    )


# ── Paired (slope) charts ────────────────────────────────────────────────────
# A ground-truth measurement shown as its actual shape (one line per subject,
# left value -> right value) rather than only the summary statistics (bias,
# spread_ratio, ...) derived from it -- e.g. Uhlrich et al.'s per-subject
# gastroc:soleus ratio, baseline vs. retention, which the transfer
# experiment's ratio-recovery stats are trying to predict.

def slope_chart_svg(pairs: list[dict[str, Any]], *, left_key: str = 'baseline',
                    right_key: str = 'retention', left_label: str = 'baseline',
                    right_label: str = 'retention', label_key: str = 'subject',
                    width: int = 320, height: int = 220) -> str:
    """`pairs`: [{label_key: ..., left_key: float, right_key: float, ...}, ...].
    Each becomes one line + two dots; a <title> carries the row's label and
    both values for hover context. Per-row overrides, all optional (default
    to a uniform currentColor line, as when every row is the same kind of
    measurement): `color`, `opacity` (line/dot opacity), `stroke_width`,
    `dash` (an SVG stroke-dasharray, e.g. '4 3', for distinguishing series
    that share a color), `flagged` (adds a 'slope-flagged' class to the row
    for the caller to style, e.g. a notable/out-of-range point)."""
    if not pairs:
        return ''
    all_vals = [p[left_key] for p in pairs] + [p[right_key] for p in pairs]
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = (y_max - y_min) or 1.0
    pad_left, pad_right, pad_top, pad_bottom = 12, 12, 16, 26
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    x_left, x_right = pad_left, pad_left + plot_w

    def _y(v: float) -> float:
        return pad_top + (1 - (v - y_min) / y_range) * plot_h

    rows_svg = []
    for p in pairs:
        y0, y1 = _y(p[left_key]), _y(p[right_key])
        title = f'{p.get(label_key, "")}: {p[left_key]:.3f} → {p[right_key]:.3f}'
        color = p.get('color', 'currentColor')
        opacity = p.get('opacity', 0.55)
        stroke_width = p.get('stroke_width', 1.5)
        dash_attr = f' stroke-dasharray="{p["dash"]}"' if p.get('dash') else ''
        row_class = 'slope-row slope-flagged' if p.get('flagged') else 'slope-row'
        rows_svg.append(
            f'<g class="{row_class}"><title>{title}</title>'
            f'<line x1="{x_left}" y1="{y0:.1f}" x2="{x_right}" y2="{y1:.1f}" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-opacity="{opacity}"'
            f'{dash_attr}/>'
            f'<circle cx="{x_left}" cy="{y0:.1f}" r="3" fill="{color}" fill-opacity="0.8"/>'
            f'<circle cx="{x_right}" cy="{y1:.1f}" r="3" fill="{color}" fill-opacity="0.8"/>'
            f'</g>'
        )

    axis_svg = (
        f'<text x="{x_left}" y="{height - 8}" text-anchor="start" '
        f'class="axis-label">{left_label}</text>'
        f'<text x="{x_right}" y="{height - 8}" text-anchor="end" '
        f'class="axis-label">{right_label}</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'class="slopechart" role="img" aria-label="{left_label} vs {right_label}">'
        + ''.join(rows_svg) + axis_svg + '</svg>'
    )


# ── Cross-stratum transfer (ratio recovery) ──────────────────────────────────
# A structurally different shape from every other metrics_doc: models[name] is
# keyed by transfer DIRECTION (base_base/base_ret/ret_ret/ret_base, plus
# pooled_base/pooled_ret -- a model trained on baseline+retention combined,
# tested on each condition separately), each a dict of metrics, plus a
# 'ratio_recovery' block of paired correlation/bias/spread-ratio stats -- not
# the flat {metric: value} shape build_models_table and build_per_output_charts
# assume. See CrossVal.ipynb's RUN_TRANSFER export cell (_DIRECTION_SOURCE)
# for the source of truth on these exact keys.
#
# ratio_recovery's sub-keys (base_trained/ret_trained/pooled_trained) are a
# DIFFERENT axis than the reconstruction directions above: each one is a
# training regime, and all of them estimate the same retention-baseline
# ratio shift -- not a cold-transfer direction of it. Keep that distinction
# in any UI copy (see _transfer_comparison.html/metrics_glossary.html).

_PREFERRED_DIRECTIONS = ('base_base', 'base_ret', 'ret_ret', 'ret_base', 'pooled_base',
                         'pooled_ret')
# ratio_recovery's sub-keys aren't transfer directions -- they're training
# regimes (what data trained the model) all estimating the SAME
# retention-baseline ratio shift, not different directions of it. See
# build_transfer_comparison's ratio_cards loop.
_PREFERRED_RATIO_CONDITIONS = ('base_trained', 'ret_trained', 'pooled_trained')
_SIGNIFICANCE_ALPHA = 0.05


def build_transfer_comparison(models: dict[str, dict]) -> dict[str, Any]:
    model_names = sorted(models.keys())

    # Prefer the known direction order; still pick up anything unexpected
    # (forward-compat, same philosophy as build_per_output_charts) rather
    # than assuming these 4 keys are the only ones that will ever exist.
    directions: list[str] = []
    for d in _PREFERRED_DIRECTIONS:
        if any(isinstance(models[m].get(d), dict) for m in model_names):
            directions.append(d)
    for m in model_names:
        for k, v in models[m].items():
            if (k not in directions and k != 'ratio_recovery'
                    and isinstance(v, dict) and 'direction' in v):
                directions.append(k)

    # Full per-direction dict, not just rrmse_w -- the template pulls
    # rrmse_w_mean/std for the table cells and 'clinical' separately (see
    # module docstring on the deliberately-generic clinical rendering).
    rrmse_table = [
        {
            'direction': d,
            'by_model': {m: (models[m].get(d) or {}) for m in model_names},
        }
        for d in directions
    ]

    ratio_cards = []
    for m in model_names:
        rr = models[m].get('ratio_recovery')
        if not isinstance(rr, dict):
            continue   # transfer not (yet) run for this architecture

        # Prefer the known condition order; still pick up anything unexpected
        # (forward-compat, same philosophy as the directions loop above)
        # rather than assuming these 3 keys are the only ones that will ever
        # exist.
        cond_names = list(_PREFERRED_RATIO_CONDITIONS)
        for k, v in rr.items():
            if k not in cond_names and k != 'transfer_cost' and isinstance(v, dict):
                cond_names.append(k)

        conditions = []
        for cond_name in cond_names:
            cond = rr.get(cond_name)
            if not isinstance(cond, dict):
                continue   # .get() in the exporter -- a condition can legitimately be None
            spearman_p = cond.get('spearman_p')
            spread_ratio = cond.get('spread_ratio') or 0
            conditions.append({
                'label': cond_name,
                **cond,
                'significant': (isinstance(spearman_p, (int, float))
                               and spearman_p < _SIGNIFICANCE_ALPHA),
                # Fixed 0..2 scale (1.0 = ideal, sits at 50% width) -- a single
                # value per condition, not a population, so no outlier-robust
                # scaling needed the way build_per_output_charts requires.
                'spread_ratio_pct': min(100.0, round(spread_ratio / 2.0 * 100, 1)),
            })
        if not conditions:
            continue

        ratio_cards.append({
            'model': m,
            'color': model_color(m, model_names),
            'conditions': conditions,
            'transfer_cost': rr.get('transfer_cost'),
        })

    return {
        'model_names': model_names,
        'directions': directions,
        'rrmse_table': rrmse_table,
        'ratio_cards': ratio_cards,
    }


# ── Per-subject ratio predictions (live) ─────────────────────────────────────
# inference.compute_gastroc_soleus_predictions runs real models against real
# test data to get ground truth + base-trained/ret-trained predicted
# gastroc:soleus ratios per subject -- this turns that into one small chart
# per subject (browsable, not just the aggregate bias/spread_ratio numbers
# above), each predicted line flagged if its subject falls outside that
# model+regime's own saved Bland-Altman limits of agreement.
#
# Only ONE model is shown per regime (the best-performing one), not every
# architecture -- with up to 4 architectures x 2 regimes, every chart was a
# 9-line tangle nobody could actually read.

def _pick_best_model(ratio_cards: list[dict[str, Any]], regime: str) -> Optional[str]:
    """The model with the highest Spearman rho for this ratio-recovery
    regime -- Spearman, not Pearson, because it's the lead/more-robust stat
    at n=10 per this site's own convention (see /metrics#ratio-recovery).
    None if no card has a usable stat for this regime."""
    best_model, best_r = None, None
    for card in ratio_cards:
        for cond in card['conditions']:
            if cond.get('label') != regime:
                continue
            r = cond.get('spearman_r')
            if isinstance(r, (int, float)) and (best_r is None or r > best_r):
                best_model, best_r = card['model'], r
    return best_model


def build_ratio_prediction_panels(predictions_data: dict[str, Any],
                                  ratio_cards: list[dict[str, Any]],
                                  model_names: list[str]) -> dict[str, Any]:
    """`predictions_data`: inference.compute_gastroc_soleus_predictions's
    return value. `ratio_cards`/`model_names`: build_transfer_comparison's --
    ratio_cards for its already-computed loa_lower/loa_upper per
    model+condition (flagging reuses those saved bounds rather than
    recomputing correlation live, which is unstable on the near-zero-
    variance predicted-shift signal these models produce) and for picking
    the best model per regime; model_names so a model's chart color here
    matches its color in the reconstruction table/ratio cards above.

    Returns {'panels': [{'subject', 'svg', 'any_flagged'}, ...],
    'best_models': {'base_trained': {'model', 'color'} or None,
    'ret_trained': {...} or None}} -- best_models is exposed once (the same
    pick applies to every subject) so the caller can render one legend above
    the whole scrollable grid instead of repeating it per panel."""
    best_model_names = {
        'base_trained': _pick_best_model(ratio_cards, 'base_trained'),
        'ret_trained': _pick_best_model(ratio_cards, 'ret_trained'),
    }
    best_models = {
        regime: {'model': name, 'color': model_color(name, model_names)} if name else None
        for regime, name in best_model_names.items()
    }

    loa_by_model_regime: dict[tuple[str, str], tuple[float, float]] = {}
    for card in ratio_cards:
        for cond in card['conditions']:
            lo, hi = cond.get('loa_lower'), cond.get('loa_upper')
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                loa_by_model_regime[(card['model'], cond['label'])] = (lo, hi)

    ground_truth = predictions_data.get('ground_truth', {})
    predictions = predictions_data.get('predictions', {})

    panels = []
    for subject in predictions_data.get('subjects', []):
        gt = ground_truth.get(subject)
        if not gt or 'baseline' not in gt or 'retention' not in gt:
            continue
        true_shift = gt['retention'] - gt['baseline']

        rows = [{**gt, 'series': 'ground truth', 'color': 'currentColor',
                'opacity': 0.9, 'stroke_width': 2.5}]
        any_flagged = False
        for regime, dash in (('base_trained', None), ('ret_trained', '4 3')):
            model_name = best_model_names[regime]
            if model_name is None:
                continue
            point = predictions.get(model_name, {}).get(regime, {}).get(subject)
            if not point or 'baseline' not in point or 'retention' not in point:
                continue
            loa = loa_by_model_regime.get((model_name, regime))
            flagged = False
            if loa is not None:
                diff = (point['retention'] - point['baseline']) - true_shift
                flagged = not (loa[0] <= diff <= loa[1])
                any_flagged = any_flagged or flagged
            row = {
                **point, 'series': f'{model_name} ({regime.replace("_", "-")})',
                'color': model_color(model_name, model_names),
                'opacity': 0.75, 'stroke_width': 1.8, 'flagged': flagged,
            }
            if dash:
                row['dash'] = dash
            rows.append(row)

        panels.append({
            'subject': subject,
            'svg': slope_chart_svg(rows, label_key='series'),
            'any_flagged': any_flagged,
        })
    return {'panels': panels, 'best_models': best_models}
