"""Schema-light access to results/metrics/*.yaml and versions/*.yaml.

The exact fields inside a metrics YAML are expected to keep changing as the
project's metric framework evolves (see notebooks/Compare_Models.ipynb and
notebooks/CrossVal.ipynb -- their two `metrics_doc` exports already disagree
on several fields). Only a small, stable envelope is read out explicitly
(each field individually optional, never a KeyError); everything else is
passed through untouched under 'models'/'extra' so new or missing fields
never require a code change here.
"""
from pathlib import Path
from typing import Any, Optional

import markdown
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO_ROOT / 'results' / 'metrics'
VERSIONS_DIR = REPO_ROOT / 'versions'
CONFIG_PATH = REPO_ROOT / 'config.yaml'
README_PATH = REPO_ROOT / 'README.md'

_RESULT_ENVELOPE_KEYS = ('version', 'eval_type', 'dataset', 'dataset_version', 'stratum', 'date',
                         'subset', 'primary_metric')
_VERSION_ENVELOPE_KEYS = ('version', 'date', 'description')


def _stringify(value: Any) -> Optional[str]:
    """YAML parses unquoted dates (e.g. `date: 2026-07-30`) as datetime.date,
    not str -- normalize so callers/schemas can always rely on a plain string."""
    return None if value is None else str(value)


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding='utf-8') as f:
            doc = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return None
    return doc if isinstance(doc, dict) else None


def _result_envelope(doc: dict, result_id: str) -> dict[str, Any]:
    models = doc.get('models')
    models = models if isinstance(models, dict) else {}
    extra = {k: v for k, v in doc.items() if k not in _RESULT_ENVELOPE_KEYS and k != 'models'}
    return {
        'id': result_id,
        'eval_type': doc.get('eval_type', 'unknown'),
        'version': doc.get('version'),
        'dataset': doc.get('dataset'),
        'dataset_version': doc.get('dataset_version'),
        'stratum': doc.get('stratum'),
        'date': _stringify(doc.get('date')),
        'subset': doc.get('subset'),
        'primary_metric': doc.get('primary_metric'),
        'model_names': sorted(models.keys()),
        'models': models,
        'extra': extra,
    }


def list_results() -> list[dict[str, Any]]:
    """All parsed results/metrics/*.yaml files.

    Missing directory -> []. Any file that fails to parse (bad YAML, or not a
    mapping at the top level) is skipped rather than failing the whole scan.
    """
    if not METRICS_DIR.is_dir():
        return []
    out = []
    for path in sorted(METRICS_DIR.glob('*.yaml')):
        doc = _load_yaml(path)
        if doc is None:
            continue
        out.append(_result_envelope(doc, path.stem))
    return out


def get_result(result_id: str) -> Optional[dict[str, Any]]:
    path = METRICS_DIR / f'{result_id}.yaml'
    if not path.is_file():
        return None
    doc = _load_yaml(path)
    if doc is None:
        return None
    return _result_envelope(doc, result_id)


def _version_envelope(doc: dict) -> dict[str, Any]:
    extra = {k: v for k, v in doc.items() if k not in _VERSION_ENVELOPE_KEYS}
    return {
        'version': doc.get('version'),
        'date': _stringify(doc.get('date')),
        'description': doc.get('description'),
        'extra': extra,
    }


def get_active_versions() -> dict[str, Any]:
    """config.yaml's `active`/`history` version pointers."""
    if not CONFIG_PATH.is_file():
        return {'active': {}, 'history': {}}
    with open(CONFIG_PATH, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return {'active': {}, 'history': {}}
    return {'active': cfg.get('active', {}) or {}, 'history': cfg.get('history', {}) or {}}


def list_datasets() -> dict[str, Any]:
    """Datasets + their known versions, derived from versions/*.yaml filenames
    (e.g. 'silder_v3.yaml' -> dataset='silder', version='3'). Generic over
    whatever dataset names show up there -- no hardcoded dataset list.
    """
    versions_by_dataset: dict[str, list[str]] = {}
    if VERSIONS_DIR.is_dir():
        for path in sorted(VERSIONS_DIR.glob('*.yaml')):
            if '_v' not in path.stem:
                continue   # e.g. training_changelog.yaml -- not a dataset version doc
            dataset, _, ver = path.stem.rpartition('_v')
            versions_by_dataset.setdefault(dataset, []).append(ver)

    active = get_active_versions()['active']
    return {
        'datasets': [
            {
                'dataset': dataset,
                'versions': sorted(vers),
                'active_version': active.get(f'{dataset}_version'),
            }
            for dataset, vers in sorted(versions_by_dataset.items())
        ],
    }


def get_dataset_version(dataset: str, version: str) -> Optional[dict[str, Any]]:
    """`version` may be a full string ('3.0') or major-only ('3') -- versions/*.yaml
    is only ever written per major version (see config.yaml's versioning convention).
    """
    major = version.split('.')[0]
    path = VERSIONS_DIR / f'{dataset}_v{major}.yaml'
    if not path.is_file():
        return None
    doc = _load_yaml(path)
    if doc is None:
        return None
    return _version_envelope(doc)


def get_readme_html() -> str:
    """README.md rendered fresh on every call -- editing the file updates the
    About page on next request, no build step or cache to invalidate."""
    if not README_PATH.is_file():
        return ''
    text = README_PATH.read_text(encoding='utf-8')
    return markdown.markdown(text, extensions=['fenced_code', 'tables'])
