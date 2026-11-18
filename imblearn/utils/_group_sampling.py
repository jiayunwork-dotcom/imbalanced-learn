"""Helpers for group-aware under-sampling.

This module implements the group selection shared by
:class:`~imblearn.under_sampling.RandomUnderSampler` and
:class:`~imblearn.ensemble.BalancedRandomForestClassifier`.

A group is an indivisible unit: all the rows carrying the same identifier are
either kept together or left out together.  The selection target of each class
is the number of rows of the smallest class in the original data, i.e. the
usual ``sampling_strategy="auto"`` target for under-sampling.
"""

# Authors: imbalanced-learn developers
# License: MIT

from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_random_state

__all__ = ["_GroupInfo", "_check_groups", "_select_groups"]


@dataclass(frozen=True)
class _GroupInfo:
    """Validated group identifiers.

    Attributes
    ----------
    codes : ndarray of shape (n_samples,)
        Integer code of the group of each row, in ``[0, n_groups)``.

    ids : ndarray of shape (n_groups,)
        Group identifiers, ordered by first appearance in the input.  The
        order defines the meaning of the codes and is deterministic for a
        given ``groups`` array.
    """

    codes: np.ndarray
    ids: np.ndarray

    @property
    def n_groups(self):
        return len(self.ids)


def _validate_group_value(value, position):
    """Return a normalised identifier or raise ``ValueError``."""
    # ``bool`` is a subclass of ``int`` but is not a meaningful group id.
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"groups[{position}] is a boolean; group identifiers must be "
            "integers or non-empty strings."
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        if len(value) == 0:
            raise ValueError(
                f"groups[{position}] is an empty string; group identifiers "
                "must be non-empty strings."
            )
        return str(value)
    raise ValueError(
        f"groups[{position}] has type {type(value).__name__}; group "
        "identifiers must be integers or non-empty strings."
    )


def _check_groups(groups, n_samples):
    """Validate ``groups`` and turn it into integer codes.

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Integer or non-empty string identifier of the group of each row.

    n_samples : int
        Expected number of rows.

    Returns
    -------
    info : :class:`_GroupInfo`
        The validated identifiers and their integer codes.
    """
    if issparse(groups):
        raise ValueError(
            "groups must be a one-dimensional array-like, got a sparse matrix."
        )

    try:
        array = np.asarray(groups, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "groups must be a one-dimensional array-like of integers or "
            "non-empty strings."
        ) from exc

    if array.ndim != 1:
        raise ValueError(
            "groups must be one-dimensional with one identifier per row; got "
            f"an array with {array.ndim} dimensions."
        )
    if len(array) != n_samples:
        raise ValueError(
            "groups and the input data must have the same number of rows; got "
            f"{len(array)} group identifiers for {n_samples} rows."
        )

    # Fast path for integer identifiers, the common case.  The identifiers
    # are ordered by first appearance so that the result does not depend on
    # the concrete values.
    raw = np.asarray(groups)
    if raw.ndim == 1 and np.issubdtype(raw.dtype, np.integer):
        raw = raw.astype(np.int64, copy=False)
        _, first_index, inverse = np.unique(
            raw, return_index=True, return_inverse=True
        )
        order = np.argsort(first_index, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        codes = rank[inverse].astype(np.intp)
        ids = raw[first_index[order]]
        return _GroupInfo(codes=codes, ids=ids)

    codes = np.empty(len(array), dtype=np.intp)
    ids = []
    positions = {}
    for position, value in enumerate(array.tolist()):
        value = _validate_group_value(value, position)
        try:
            code = positions[value]
        except KeyError:
            code = len(ids)
            positions[value] = code
            ids.append(value)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"groups[{position}] is not a valid group identifier."
            ) from exc
        codes[position] = code

    return _GroupInfo(codes=codes, ids=np.asarray(ids, dtype=object))


def _select_groups(y, codes, n_groups, random_state=None):
    """Select whole groups so that the class counts approach the target.

    The target of every class is the size of the smallest class in ``y``.  The
    trade-off between the classes touched by a mixed group is measured by the
    total L1 deviation between the counts of the selected rows and the target
    of every class: a group is removed when it brings the sum of the absolute
    deviations strictly down.

    Two hard constraints are always enforced:

    * every class of the input must remain represented (at least one row),
    * a group is left out whenever doing so keeps every class represented.

    Groups are visited best-first: at each step the group whose removal lowers
    the deviation the most is dropped, ties being broken uniformly at random.
    For single-label groups of equal size this rule reaches an exact balance
    whenever one exists.  The randomised tie-breaking gives different trees
    different -- but equally good -- group selections.

    The working set is the ``n_groups x n_classes`` count table, never a
    ``n_groups x n_samples`` matrix.  When every group carries a single label
    the deviation is separable by class and the selection is solved from the
    per-class group sizes in ``O(n_samples + P)``, with ``P`` the number of
    pure groups: equal sizes need no search at all and heterogeneous sizes
    consume a size histogram.  Mixed groups share their class-count vector and
    are therefore grouped into a handful of distinct signatures, on which the
    best-first removal runs.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class label of each row.

    codes : ndarray of shape (n_samples,)
        Group code of each row.

    n_groups : int
        Number of distinct groups.

    random_state : int, RandomState instance or None, default=None
        Controls the random tie-breaking.

    Returns
    -------
    selected_mask : ndarray of shape (n_groups,) of bool
        Whether each group is selected.

    counts : ndarray of shape (n_groups, n_classes)
        Number of rows of each class in each group.

    target : int
        Balanced target, i.e. the number of rows of the smallest class.
    """
    rng = check_random_state(random_state)

    classes, y_codes = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    counts = np.zeros((n_groups, n_classes), dtype=np.int64)
    np.add.at(counts, (codes, y_codes), 1)
    totals = counts.sum(axis=0)
    target = int(totals.min())

    if np.all((counts > 0).sum(axis=1) == 1):
        return _select_pure_groups(counts, target, rng)

    return _select_mixed_groups(counts, target, rng)


def _select_pure_groups(counts, target, rng):
    """Select pure groups without a global subset enumeration.

    The problem separates by class: the deviation of one class only depends on
    the groups of that class.  When all the groups of a class have the same
    size ``s`` the best possible count is the multiple of ``s`` closest to the
    target, so the number of kept groups is computed directly and a uniformly
    random subset of that size is selected.  This is exact whenever a balanced
    count is reachable.

    With heterogeneous group sizes the size histogram is consumed best-first:
    the removal whose size is closest to the remaining distance is taken, so a
    removal costs the number of distinct sizes of the class.  The selection is a
    good local optimum rather than a global one, which is enough for the
    sampling use case.  Both paths are linear or near-linear in the number of
    groups of the class.
    """
    n_groups, n_classes = counts.shape
    selected = np.ones(n_groups, dtype=bool)

    for label in range(n_classes):
        members = np.flatnonzero(counts[:, label] > 0)
        sizes = counts[members, label]
        if members.size <= 1:
            continue
        present = int(sizes.sum())
        if present <= target:
            continue

        if np.all(sizes == sizes[0]):
            size = int(sizes[0])
            # Choose the number of groups kept for this class: any multiple of
            # ``size`` between ``size`` and ``present`` is reachable.  Ties are
            # broken uniformly at random.
            candidates = np.arange(1, members.size + 1)
            distances = np.abs(candidates * size - target)
            best = candidates[distances == distances.min()]
            kept = int(best[rng.randint(best.size)])
            if kept == members.size:
                continue
            chosen = rng.permutation(members.size)[:kept]
            dropped = np.ones(members.size, dtype=bool)
            dropped[chosen] = False
            selected[members[dropped]] = False
            continue

        # Heterogeneous sizes: best-first removal using the size histogram.
        # The deviation of the class only depends on the group size, so all
        # groups of a given size are interchangeable; one of them is picked
        # uniformly at random.  Each bucket is consumed from the end so that a
        # removal is O(1).
        histogram = {}
        for group, size in zip(members.tolist(), sizes.tolist()):
            histogram.setdefault(int(size), []).append(int(group))
        available = {size: len(groups) for size, groups in histogram.items()}
        present = int(sizes.sum())
        kept = members.size
        while kept > 1 and present > target:
            distance = present - target
            best_size = None
            best_cost = distance
            for size in histogram:
                if available[size] == 0:
                    continue
                cost = abs(distance - size)
                if cost < best_cost:
                    best_cost = cost
                    best_size = size
            if best_size is None:
                break
            groups = histogram[best_size]
            index = rng.randint(available[best_size])
            group = groups[index]
            groups[index] = groups[available[best_size] - 1]
            available[best_size] -= 1
            selected[group] = False
            present -= best_size
            kept -= 1

    if selected.all():
        # Nothing was held out but coverage allows it: drop the least
        # damaging group, as required for the out-of-bag estimate.
        group = _hold_out_least_damaging(counts, target, rng)
        if group is not None:
            selected[group] = False

    return selected, counts, target


def _hold_out_least_damaging(counts, target, rng):
    """Return a group that can be dropped with the smallest L1 increase."""
    totals = counts.sum(axis=0)
    # A group can be dropped if every class keeps at least one row.
    removable = np.flatnonzero(np.all(totals - counts >= 1, axis=1))
    if removable.size == 0:
        return None
    damages = np.abs(totals - counts[removable] - target).sum(
        axis=1
    ) - np.abs(totals - target).sum()
    best = int(damages.min())
    tied = removable[damages == best]
    return int(tied[rng.randint(tied.size)])


def _select_mixed_groups(counts, target, rng):
    """Best-first selection when some group carries several labels.

    Groups with the same class-count vector are interchangeable, so the
    best-first search runs on the distinct vectors rather than on the groups.
    The number of distinct vectors is bounded by ``(rows_per_group + 1) **
    n_classes`` and is usually tiny.  Each search step shows every remaining
    vector, then removes one group of the winning vector, picked uniformly at
    random; equivalent groups therefore end up in different trees across the
    forest.
    """
    n_groups, n_classes = counts.shape
    signatures, inverse, multiplicities = np.unique(
        counts, axis=0, return_inverse=True, return_counts=True
    )
    # Group ids of each signature, in a random order so that the groups that
    # are actually dropped differ between trees.
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(signatures) + 1))
    buckets = []
    for signature_index in range(len(signatures)):
        bucket = order[bounds[signature_index] : bounds[signature_index + 1]]
        rng.shuffle(bucket)
        # A consumption pointer is enough: consumed entries stay at the front.
        buckets.append(bucket)

    remaining = multiplicities.astype(np.int64, copy=True)
    consumed = np.zeros(len(signatures), dtype=np.int64)
    selected = np.ones(n_groups, dtype=bool)
    current = signatures.T @ multiplicities

    def deviation(state):
        return int(np.abs(state - target).sum())

    removed_any = False
    while True:
        available = np.flatnonzero(remaining > 0)
        if available.size == 0:
            break
        # Removing a group must leave at least one row of every class.
        feasible = available[
            np.all(current - signatures[available] >= 1, axis=1)
        ]
        if feasible.size == 0:
            break

        current_cost = deviation(current)
        costs = np.abs(current - signatures[feasible] - target).sum(axis=1)
        best = int(costs.min())

        if best < current_cost:
            tied = feasible[costs == best]
        elif not removed_any:
            # Nothing improves the balance, but at least one group must be
            # held out whenever the remaining ones still cover every class.
            # Drop the least damaging one.
            tied = feasible[costs == best]
        else:
            break

        signature_index = int(tied[rng.randint(tied.size)])
        group = int(buckets[signature_index][consumed[signature_index]])
        consumed[signature_index] += 1
        remaining[signature_index] -= 1
        selected[group] = False
        current = current - signatures[signature_index]
        removed_any = True

    return selected, counts, target
