"""Envelope-only response models.

Only the small, stable set of fields every metrics_doc is expected to carry
(id, eval_type, dataset, dataset_version, stratum, date) is strictly typed.
The actual per-model metrics -- which are still actively evolving across
notebooks/Compare_Models.ipynb and notebooks/CrossVal.ipynb -- are passed
through as plain dicts (`models`, `extra`) rather than broken out field by
field, so adding/removing a metric in a notebook never requires an API change.
"""
from typing import Any, Optional

from pydantic import BaseModel


class ResultSummary(BaseModel):
    id: str
    eval_type: str
    version: Optional[str] = None
    dataset: Optional[str] = None
    dataset_version: Optional[str] = None
    stratum: Optional[str] = None
    date: Optional[str] = None
    model_names: list[str] = []


class ResultDetail(ResultSummary):
    models: dict[str, Any] = {}
    extra: dict[str, Any] = {}


class DatasetVersionInfo(BaseModel):
    version: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None
    extra: dict[str, Any] = {}


class DatasetInfo(BaseModel):
    dataset: str
    versions: list[str] = []
    active_version: Optional[str] = None


class DatasetsResponse(BaseModel):
    datasets: list[DatasetInfo] = []
