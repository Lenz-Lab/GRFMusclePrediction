import numpy as np
import pytest

from grf_pipeline_utils.data_utils import (
    exclude_segments,
    filter_segments,
    filter_segs_by_metadata,
    flatten_to_muscle_dict,
    get_all_segments,
    interp_segments,
    normalize_by_mass_in_order,
    pack_metadata,
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


# ── pack_metadata ───────────────────────────────────────────────────────────────

def _dict_to_array_rowcount(split_dict, all_keys):
    """
    Minimal stand-in for the split notebooks' dict_to_array (which packs
    ALL_KEYS into the X/y array) -- lives only in Split_Single_Dataset.ipynb /
    Split_Multiple_Datasets.ipynb, not importable here. Only the row count is
    needed to test the alignment guarantee pack_metadata depends on.
    """
    total = 0
    for subj, data in split_dict.items():
        total += len(data[all_keys[0]])
    return total


def _make_split_dict(n_subjects=2, n_segs=3, extra_keys=('trial_name',)):
    """Build a synthetic split_dict with one numeric signal plus arbitrary
    per-segment metadata keys, so tests don't hardcode 'trial_name' as the
    only possible groupable key."""
    split_dict = {}
    for i in range(n_subjects):
        subj = f'S{i + 1}'
        split_dict[subj] = {'grf_x': [np.ones(10) for _ in range(n_segs)]}
        for key in extra_keys:
            split_dict[subj][key] = [f'{key}_{subj}_{j}' for j in range(n_segs)]
    return split_dict


def test_pack_metadata_subject_id_matches_dict_keys():
    """subject_id should repeat each subject's key once per its segments."""
    split_dict = _make_split_dict(n_subjects=2, n_segs=3)
    result = pack_metadata(split_dict, metadata_keys=['trial_name'])
    assert list(result['subject_id']) == ['S1', 'S1', 'S1', 'S2', 'S2', 'S2']


def test_pack_metadata_values_match_source_order():
    """Packed values for a metadata key should match the source lists in order."""
    split_dict = _make_split_dict(n_subjects=2, n_segs=2, extra_keys=('trial_name',))
    result = pack_metadata(split_dict, metadata_keys=['trial_name'])
    assert list(result['trial_name']) == [
        'trial_name_S1_0', 'trial_name_S1_1',
        'trial_name_S2_0', 'trial_name_S2_1',
    ]


@pytest.mark.parametrize('metadata_keys', [
    ['trial_name'],
    ['trial_name', 'side'],
    ['trial_name', 'side', 'speed'],
])
def test_pack_metadata_arbitrary_key_lists(metadata_keys):
    """
    pack_metadata must work for any list of metadata key names, not just
    today's 'trial_name' -- proves adding a new groupable key later (e.g.
    'side' or a separately-parsed 'speed') requires no changes to this
    function, only adding the key's name to the caller's metadata_keys list.
    """
    split_dict = _make_split_dict(n_subjects=2, n_segs=3, extra_keys=tuple(metadata_keys))
    result = pack_metadata(split_dict, metadata_keys=metadata_keys)

    assert set(result.keys()) == {'subject_id', *metadata_keys}
    for key in metadata_keys:
        assert len(result[key]) == 6  # 2 subjects * 3 segs
    assert len(result['subject_id']) == 6


def test_pack_metadata_row_count_matches_dict_to_array():
    """
    The row count pack_metadata produces must equal what the split notebooks'
    dict_to_array would produce for the same split_dict -- this is the actual
    alignment guarantee the whole subject_id/trial_name design depends on, so
    it's asserted directly rather than just implied by matching loop structure.
    """
    split_dict = _make_split_dict(n_subjects=3, n_segs=4, extra_keys=('trial_name',))
    expected_rows = _dict_to_array_rowcount(split_dict, all_keys=['grf_x'])

    result = pack_metadata(split_dict, metadata_keys=['trial_name'])
    assert len(result['subject_id']) == expected_rows
    assert len(result['trial_name']) == expected_rows


# ── filter_segments: generic list-key passthrough ───────────────────────────────

def _make_segs_with_trial_name():
    """
    2 subjects, each with 4 segments tagged 'baseline'/'baseline'/'retention'/'retention',
    plus 'time_resampled' as a non-dict entry that should pass through untouched.
    """
    segs = {
        'S1': {
            'grf_y':      [np.ones(10) * 1, np.ones(10) * 2, np.ones(10) * 3, np.ones(10) * 4],
            'trial_name': ['baseline', 'baseline', 'retention', 'retention'],
        },
        'S2': {
            'grf_y':      [np.ones(10) * 5, np.ones(10) * 6, np.ones(10) * 7, np.ones(10) * 8],
            'trial_name': ['baseline', 'retention', 'retention', 'retention'],
        },
        'time_resampled': [np.linspace(0, 1, 10)],
    }
    return segs


# ── filter_segs_by_metadata ─────────────────────────────────────────────────────

def test_filter_segs_by_metadata_basic_filtering():
    """Should keep only segments whose trial_name is in the requested values."""
    segs = _make_segs_with_trial_name()
    result = filter_segs_by_metadata(segs, 'trial_name', ['baseline'])
    assert result['S1']['trial_name'] == ['baseline', 'baseline']
    assert result['S2']['trial_name'] == ['baseline']


def test_filter_segs_by_metadata_keeps_signals_aligned():
    """Filtering must apply identically to every list-valued signal, not just the key."""
    segs = _make_segs_with_trial_name()
    result = filter_segs_by_metadata(segs, 'trial_name', ['retention'])
    # S1's retention segments were grf_y values 3 and 4
    assert np.allclose(result['S1']['grf_y'][0], 3.0)
    assert np.allclose(result['S1']['grf_y'][1], 4.0)
    assert len(result['S1']['grf_y']) == len(result['S1']['trial_name']) == 2


def test_filter_segs_by_metadata_subject_with_zero_matches_kept_empty():
    """A subject with no matching segments should be kept with empty lists, not dropped."""
    segs = _make_segs_with_trial_name()
    result = filter_segs_by_metadata(segs, 'trial_name', ['nonexistent_stratum'])
    assert 'S1' in result and 'S2' in result
    assert result['S1']['grf_y'] == []
    assert result['S1']['trial_name'] == []


def test_filter_segs_by_metadata_missing_key_treated_as_no_matches():
    """A subject lacking the metadata key entirely should come back empty, not error."""
    segs = _make_segs_with_trial_name()
    segs['S3'] = {'grf_y': [np.ones(10)]}   # no 'trial_name' key
    result = filter_segs_by_metadata(segs, 'trial_name', ['baseline'])
    assert result['S3']['grf_y'] == []


def test_filter_segs_by_metadata_preserves_non_dict_entries():
    """Non-subject entries like 'time_resampled' should pass through unchanged."""
    segs = _make_segs_with_trial_name()
    result = filter_segs_by_metadata(segs, 'trial_name', ['baseline'])
    assert result['time_resampled'] is segs['time_resampled']


def test_filter_segs_by_metadata_does_not_mutate_input():
    """The input dict should be left untouched -- a new dict is returned."""
    segs = _make_segs_with_trial_name()
    original_len = len(segs['S1']['grf_y'])
    filter_segs_by_metadata(segs, 'trial_name', ['baseline'])
    assert len(segs['S1']['grf_y']) == original_len


def test_filter_segments_passes_through_arbitrary_metadata_key():
    """
    filter_segments must filter ANY list-valued key of matching length in
    lockstep with the muscle-based keep_mask -- not just known signal names.
    This is the invariant every future groupable metadata key (trial_name
    today, anything added later) relies on to survive filtering un-desynced.
    Uses a key name ('grouping_key') that isn't a real signal, to prove
    nothing here is special-cased to 'trial_name' specifically.
    """
    n_good = 20
    good_segs  = [np.ones(50) * 1.0 for _ in range(n_good)]
    bad_seg    = [np.ones(50) * 100.0]  # far outside the band -> should be dropped
    good_tags  = [f'good_{i}' for i in range(n_good)]
    bad_tag    = ['bad_outlier']

    seg_dict = {
        'S1': {
            'm1':            good_segs + bad_seg,
            'grouping_key':  good_tags + bad_tag,
        }
    }

    filtered, dropped, bands = filter_segments(seg_dict, muscle_keys=['m1'])

    assert len(filtered['S1']['m1']) == n_good
    assert filtered['S1']['grouping_key'] == good_tags
    assert 'bad_outlier' not in filtered['S1']['grouping_key']
