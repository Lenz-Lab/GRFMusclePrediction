import numpy as np
import pytest

from grf_pipeline_utils.data_utils import (
    exclude_segments,
    flatten_to_muscle_dict,
    get_all_segments,
    interp_segments,
    normalize_by_mass_in_order,
)

# ── interp_segments ────────────────────────────────────────────────────────────

def test_interp_output_length():
    """All resampled segments should have exactly n_interp_points timepoints."""
    segments = [np.random.rand(47), np.random.rand(83), np.random.rand(61)]
    resampled, _ = interp_segments(segments, n_interp_points=100)
    for seg in resampled:
        assert len(seg) == 100


def test_interp_preserves_endpoints():
    """Endpoint values should be preserved after interpolation."""
    seg = np.array([0.0, 0.5, 1.0])
    resampled, _ = interp_segments([seg], n_interp_points=10)
    assert np.isclose(resampled[0][0],  0.0, atol=1e-6)
    assert np.isclose(resampled[0][-1], 1.0, atol=1e-6)


def test_interp_returns_time_vector():
    """Time vectors should be in [0, 1] and have correct length."""
    segments = [np.random.rand(50)]
    _, time = interp_segments(segments, n_interp_points=100)
    assert len(time[0]) == 100
    assert np.isclose(time[0][0],  0.0, atol=1e-6)
    assert np.isclose(time[0][-1], 1.0, atol=1e-6)


def test_interp_constant_segment():
    """A constant segment should remain constant after interpolation."""
    seg = np.ones(30) * 5.0
    resampled, _ = interp_segments([seg], n_interp_points=100)
    assert np.allclose(resampled[0], 5.0, atol=1e-6)


def test_interp_multiple_segments_different_lengths():
    """Segments of varying lengths should all resample to the same target length."""
    lengths = [30, 50, 75, 110]
    segments = [np.random.rand(n) for n in lengths]
    resampled, _ = interp_segments(segments, n_interp_points=100)
    assert len(resampled) == 4
    assert all(len(s) == 100 for s in resampled)


# ── exclude_segments ───────────────────────────────────────────────────────────

def test_exclude_removes_short_segments():
    """Segments shorter than min_len should be removed."""
    segments = [np.ones(10), np.ones(50), np.ones(100)]
    result = exclude_segments(segments, min_len=20, max_len=200)
    assert len(result) == 2
    assert all(len(s) >= 20 for s in result)


def test_exclude_removes_long_segments():
    """Segments longer than max_len should be removed."""
    segments = [np.ones(50), np.ones(100), np.ones(200)]
    result = exclude_segments(segments, min_len=10, max_len=150)
    assert len(result) == 2
    assert all(len(s) <= 150 for s in result)


def test_exclude_keeps_segments_within_range():
    """Segments within range should all be kept."""
    segments = [np.ones(n) for n in [40, 60, 80, 100]]
    result = exclude_segments(segments, min_len=40, max_len=100)
    assert len(result) == 4


def test_exclude_empty_result():
    """If all segments are out of range, result should be empty."""
    segments = [np.ones(5), np.ones(10)]
    result = exclude_segments(segments, min_len=50, max_len=200)
    assert result == []


def test_exclude_boundary_values():
    """Segments exactly at min_len and max_len should be kept."""
    segments = [np.ones(50), np.ones(100)]
    result = exclude_segments(segments, min_len=50, max_len=100)
    assert len(result) == 2


# ── normalize_by_mass_in_order ─────────────────────────────────────────────────

def _make_seg_dict(n_subjects=3, n_segs=5, seg_len=100):
    """Helper to build a minimal seg_dict for testing."""
    keys_to_norm = ['grf_y', 'achilles']
    seg_dict = {}
    for i in range(n_subjects):
        subj = f'S{i+1}'
        seg_dict[subj] = {
            'grf_y':    [np.ones(seg_len) * 100.0 for _ in range(n_segs)],
            'achilles': [np.ones(seg_len) * 200.0 for _ in range(n_segs)],
            'grf_x':    [np.ones(seg_len) * 50.0  for _ in range(n_segs)],
        }
    return seg_dict, keys_to_norm


def test_normalize_divides_by_mass():
    """Normalized values should equal original divided by mass."""
    seg_dict, keys_to_norm = _make_seg_dict(n_subjects=2)
    masses = {'S1': 70.0, 'S2': 80.0}
    result = normalize_by_mass_in_order(seg_dict, masses, keys_to_norm)

    assert np.allclose(result['S1']['grf_y'][0],    100.0 / 70.0, atol=1e-6)
    assert np.allclose(result['S2']['grf_y'][0],    100.0 / 80.0, atol=1e-6)
    assert np.allclose(result['S1']['achilles'][0], 200.0 / 70.0, atol=1e-6)


def test_normalize_leaves_other_keys_unchanged():
    """Keys not in keys_to_normalize should be untouched."""
    seg_dict, keys_to_norm = _make_seg_dict(n_subjects=2)
    masses = {'S1': 70.0, 'S2': 80.0}
    result = normalize_by_mass_in_order(seg_dict, masses, keys_to_norm)

    assert np.allclose(result['S1']['grf_x'][0], 50.0, atol=1e-6)
    assert np.allclose(result['S2']['grf_x'][0], 50.0, atol=1e-6)


def test_normalize_raises_on_mass_mismatch():
    """Should raise ValueError if a subject has no mass entry."""
    seg_dict, keys_to_norm = _make_seg_dict(n_subjects=3)
    masses = {'S1': 70.0, 'S2': 80.0}  # S3 missing
    with pytest.raises(ValueError):
        normalize_by_mass_in_order(seg_dict, masses, keys_to_norm)


def test_normalize_preserves_subject_count():
    """Output should have the same number of subjects as input."""
    seg_dict, keys_to_norm = _make_seg_dict(n_subjects=4)
    masses = {'S1': 60.0, 'S2': 70.0, 'S3': 80.0, 'S4': 90.0}
    result = normalize_by_mass_in_order(seg_dict, masses, keys_to_norm)
    subject_keys = [k for k, v in result.items() if isinstance(v, dict)]
    assert len(subject_keys) == 4


# ── get_all_segments ───────────────────────────────────────────────────────────

def test_get_all_segments_correct_count():
    """Should return all segments for a key across all subjects."""
    seg_dict = {
        'S1': {'grf_y': [np.ones(100), np.ones(100)]},
        'S2': {'grf_y': [np.ones(100), np.ones(100), np.ones(100)]},
    }
    result = get_all_segments(seg_dict, 'grf_y')
    assert result.shape[0] == 5


def test_get_all_segments_skips_time_resampled():
    """The 'time_resampled' key should be ignored."""
    seg_dict = {
        'S1':            {'grf_y': [np.ones(100)]},
        'time_resampled': np.linspace(0, 1, 100),
    }
    result = get_all_segments(seg_dict, 'grf_y')
    assert result.shape[0] == 1


# ── flatten_to_muscle_dict ─────────────────────────────────────────────────────

def test_flatten_correct_shape():
    """Should stack all segments for each muscle across subjects."""
    seg_dict = {
        'S1': {'achilles': [np.ones(100), np.ones(100)]},
        'S2': {'achilles': [np.ones(100)]},
    }
    result = flatten_to_muscle_dict(seg_dict, muscle_keys=['achilles'])
    assert result['achilles'].shape == (3, 100)


def test_flatten_missing_muscle_skipped():
    """Subjects missing a muscle key should be handled without error."""
    seg_dict = {
        'S1': {'achilles': [np.ones(100)]},
        'S2': {'tibant':   [np.ones(100)]},
    }
    result = flatten_to_muscle_dict(seg_dict, muscle_keys=['achilles', 'tibant'])
    assert result['achilles'].shape[0] == 1
    assert result['tibant'].shape[0] == 1
