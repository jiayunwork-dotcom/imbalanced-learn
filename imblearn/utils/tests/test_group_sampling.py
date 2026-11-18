"""Tests for the group-aware under-sampling helpers."""

# Authors: imbalanced-learn developers
# License: MIT

from itertools import combinations

import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal

from imblearn.utils._group_sampling import _check_groups, _select_groups


def _brute_force_optimum(counts, target, hold_out_required):
    """Best reachable L1 deviation under the coverage and hold-out rules."""
    n_groups = len(counts)
    best = np.inf
    for n_kept in range(1, n_groups + 1):
        if hold_out_required and n_kept == n_groups:
            continue
        for kept in combinations(range(n_groups), n_kept):
            selected_counts = counts[list(kept)].sum(axis=0)
            if np.all(selected_counts >= 1):
                best = min(best, int(np.abs(selected_counts - target).sum()))
    return best


def _assert_valid_selection(y, codes, n_groups, selected, counts, target):
    assert selected.shape == (n_groups,)
    selected_counts = counts[selected].sum(axis=0)
    # Every class is represented.
    assert np.all(selected_counts >= 1)
    # Each group is in or out, so each row is selected at most once.
    assert selected_counts.sum() <= len(y)
    # A group must be held out whenever the remaining ones cover every class.
    totals = counts.sum(axis=0)
    can_hold = any(np.all(totals - counts[g] >= 1) for g in range(n_groups))
    if can_hold:
        assert not selected.all()
    return selected_counts


def test_select_groups_equal_size_single_label_exact_balance():
    # Class 0: 3 groups, class 1: 2 groups, class 2: 4 groups of 2 rows.
    y = np.array([0] * 6 + [1] * 4 + [2] * 8)
    codes = np.repeat(np.arange(9), 2)
    for seed in range(20):
        selected, counts, target = _select_groups(y, codes, 9, random_state=seed)
        selected_counts = _assert_valid_selection(
            y, codes, 9, selected, counts, target
        )
        assert target == 4
        assert_array_equal(selected_counts, np.array([4, 4, 4]))
        # Groups are selected in a random order and differ across seeds.
        if seed == 0:
            first = tuple(sorted(np.flatnonzero(selected).tolist()))
        else:
            selections = {first, tuple(sorted(np.flatnonzero(selected).tolist()))}
    assert len(selections) > 1


@pytest.mark.parametrize("seed", range(10))
def test_select_groups_heterogeneous_sizes_match_brute_force(seed):
    rng = np.random.RandomState(seed)
    n_groups = rng.randint(3, 8)
    n_classes = rng.randint(2, 4)
    labels = rng.randint(0, n_classes, n_groups)
    if np.unique(labels).size < n_classes:
        pytest.skip("a class is missing from the drawn groups")
    sizes = rng.randint(1, 4, n_groups)
    y = np.repeat(labels, sizes)
    codes = np.repeat(np.arange(n_groups), sizes)

    selected, counts, target = _select_groups(
        y, codes, n_groups, random_state=seed
    )
    selected_counts = _assert_valid_selection(
        y, codes, n_groups, selected, counts, target
    )
    optimum = _brute_force_optimum(
        counts, target, hold_out_required=_can_hold(counts)
    )
    # Heterogeneous group sizes use a best-first heuristic: it is not a global
    # optimum, but the deviation stays within a small number of rows of it.
    assert int(np.abs(selected_counts - target).sum()) <= optimum + 2


def _can_hold(counts):
    totals = counts.sum(axis=0)
    return any(np.all(totals - counts[g] >= 1) for g in range(len(counts)))


@pytest.mark.parametrize("seed", range(10))
def test_select_groups_mixed_labels_match_brute_force(seed):
    rng = np.random.RandomState(seed)
    n_samples = rng.randint(6, 16)
    n_classes = rng.randint(2, 4)
    y = rng.randint(0, n_classes, n_samples)
    if np.unique(y).size < n_classes:
        pytest.skip("a class is missing from the drawn rows")
    n_groups = rng.randint(2, min(6, n_samples))
    codes = rng.randint(0, n_groups, n_samples)
    # Compact the group codes.
    _, codes = np.unique(codes, return_inverse=True)
    n_groups = codes.max() + 1

    selected, counts, target = _select_groups(
        y, codes, n_groups, random_state=seed
    )
    selected_counts = _assert_valid_selection(
        y, codes, n_groups, selected, counts, target
    )
    hold_out_required = _can_hold(counts)
    optimum = _brute_force_optimum(counts, target, hold_out_required)
    if not hold_out_required:
        optimum = min(optimum, int(np.abs(counts.sum(axis=0) - target).sum()))
    assert int(np.abs(selected_counts - target).sum()) <= optimum


def test_select_groups_mixed_labels_visible_deviation():
    # Group 0: four rows of class 0; group 1: four rows of class 1; group 2:
    # one row of each class; group 3: two rows of each class.  Both classes
    # have 7 rows, so the target is 7.  Keeping everything is balanced, but a
    # group must be held out: the best reachable counts are 6/6 or 5/5, i.e.
    # both classes land 1 or 2 rows below the target.
    y = np.array([0] * 4 + [1] * 4 + [0, 1] + [0, 0, 1, 1])
    codes = np.repeat(np.arange(4), [4, 4, 2, 4])
    selected, counts, target = _select_groups(y, codes, 4, random_state=0)
    assert target == 7
    selected_counts = counts[selected].sum(axis=0)
    assert not selected.all()
    assert np.all(selected_counts >= 1)
    # The deviation is visible in the actual class counts and matches the
    # optimal hold-out-constrained solution.
    assert int(np.abs(selected_counts - target).sum()) == 2
    assert_array_equal(np.sort(selected_counts), np.array([6, 6]))
    assert not selected[2]


def test_select_groups_no_group_left_out_when_coverage_required():
    # Two groups of the same class only: dropping either one loses the class.
    y = np.array([0, 0, 1, 1])
    codes = np.array([0, 0, 1, 1])
    selected, counts, target = _select_groups(y, codes, 2, random_state=0)
    assert selected.all()
    assert_array_equal(counts[selected].sum(axis=0), np.array([2, 2]))


def test_select_groups_single_group():
    y = np.array([0, 1, 0, 1])
    selected, counts, target = _select_groups(
        y, np.zeros(4, dtype=int), 1, random_state=0
    )
    assert_array_equal(selected, np.array([True]))
    assert target == 2


def test_group_info_codes_follow_first_appearance():
    groups = np.array([7, 3, 7, 9, 3])
    info = _check_groups(groups, 5)
    assert_array_equal(info.codes, np.array([0, 1, 0, 2, 1]))
    assert_array_equal(info.ids, np.array([7, 3, 9]))

    strings = np.array(["b", "a", "b", "c", "a"])
    info = _check_groups(strings, 5)
    assert_array_equal(info.codes, np.array([0, 1, 0, 2, 1]))
    assert_array_equal(info.ids, np.array(["b", "a", "c"]))


def test_group_info_accepts_lists_and_mixed_values():
    info = _check_groups([1, "a", 1, "a"], 4)
    assert_array_equal(info.codes, np.array([0, 1, 0, 1]))
    assert_array_equal(info.ids, np.array([1, "a"], dtype=object))


@pytest.mark.parametrize(
    "groups, match",
    [
        (np.zeros(3, dtype=int), "same number of rows"),
        (np.zeros((4, 2), dtype=int), "one-dimensional"),
        (np.array([None] * 4), "NoneType"),
        (np.array([1.5] * 4), "type float"),
        (np.array(["a", "b", "", ""]), "empty string"),
        (np.array([True] * 4), "boolean"),
    ],
)
def test_check_groups_errors(groups, match):
    with pytest.raises(ValueError, match=match):
        _check_groups(groups, 4)
