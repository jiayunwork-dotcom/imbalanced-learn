"""Shared planning helpers for over-samplers that generate new rows.

Some over-samplers do not generate samples from the class as a whole but from
several *sources*: the anchors of :class:`~imblearn.over_sampling.ADASYN` or
the eligible clusters of :class:`~imblearn.over_sampling.KMeansSMOTE`.  Each
sampler decides on its own which sources are eligible and how much relative
weight each of them carries; the bookkeeping that turns those weights into an
exact integer number of new rows per source is shared and lives here, so that
the two samplers produce the same per-source plan from the same ordered
weights, the same requested total and the same integer seed.

The allocation rule
-------------------
Given ``k`` ordered non-negative weights ``w`` and a total ``n`` of rows to
generate, the *ideal share* of source ``i`` is::

    ideal_i = n * w_i / sum(w)

The plan gives source ``i`` either ``floor(ideal_i)`` or ``ceil(ideal_i)``
rows, so that the number of rows it actually receives differs from its ideal
share by strictly less than one row (up to floating-point rounding).  Zero
weights are never rounded up: a source that receives no weight receives no
row, even when ``n`` is small.  Consequently, when there are fewer rows to
generate than sources with a positive weight, some sources with a positive
share receive no row at all (their ideal share is below one).

The total is exact by construction: first every source receives
``floor(ideal_i)``; the ``n - sum(floor(ideal_i))`` remaining rows are then
handed out to the sources with the largest fractional parts (the *largest
remainder* method).  There is no rounding of the total, hence no over- or
under-generation.

Tie-breaking
------------
Fractional parts are frequently tied, in particular when all weights are
equal.  Breaking those ties by position would systematically favour the
sources that happen to come first, which is undesirable for a plan that is
supposed to be neutral.  Ties are therefore broken by a random permutation
drawn from ``random_state``.  An integer seed makes the permutation, and
hence the plan, reproducible: identical weights, total and seed always give
identical per-source counts.  With a non-integer, non-fixed ``random_state``
the plan is drawn from a shared random stream and is not reproducible.

The algorithm needs ``O(k)`` additional memory (the weights, the floors and
the tie order); it never materialises a source-by-row table.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import operator

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["allocate_integer_quotas", "neighbors_excluding_self"]


def allocate_integer_quotas(weights, n_samples, random_state=None):
    """Split ``n_samples`` rows over weighted sources as integers.

    The returned counts sum exactly to ``n_samples``.  Every source with a
    positive weight receives ``floor(ideal)`` or ``ceil(ideal)`` rows, where
    ``ideal = n_samples * weight / sum(weights)``; sources with a zero weight
    receive zero.  The remaining rows are assigned by largest fractional
    remainder, ties being broken by a permutation drawn from
    ``random_state`` (see the module docstring for the rationale).

    Parameters
    ----------
    weights : array-like of shape (n_sources,)
        Non-negative, finite relative weights in source order.  The weights
        need not sum to one.

    n_samples : int
        Exact number of rows to distribute.  It must be non-negative.

    random_state : int, RandomState instance or None, default=None
        Source of randomness used to break ties between equal fractional
        parts.  Pass an integer for a reproducible plan.

    Returns
    -------
    quotas : ndarray of shape (n_sources,), dtype=int64
        Integer number of rows allocated to each source, in the same order as
        ``weights``.

    Raises
    ------
    ValueError
        If ``weights`` is not one-dimensional, contains negative or
        non-finite values, if ``n_samples`` is negative, or if rows are
        requested while no source has a positive weight.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(
            "weights must be one-dimensional. Got an array with "
            f"{weights.ndim} dimensions instead."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("weights cannot contain negative values.")
    try:
        n_samples = operator.index(n_samples)
    except TypeError as exc:
        raise ValueError(
            f"n_samples must be an integer. Got {n_samples!r} instead."
        ) from exc
    if n_samples < 0:
        raise ValueError("n_samples cannot be negative.")

    if weights.size == 0 and n_samples > 0:
        raise ValueError(
            "Cannot allocate samples: no source is available to generate them."
        )
    quotas = np.zeros(weights.size, dtype=np.int64)
    if n_samples == 0:
        return quotas

    total_weight = weights.sum()
    if not total_weight > 0:
        raise ValueError(
            "Cannot allocate samples: at least one weight must be strictly positive."
        )

    ideal = weights * (n_samples / total_weight)
    quotas = np.floor(ideal).astype(np.int64)
    # ``sum(floor(ideal)) <= n_samples`` in exact arithmetic because
    # ``sum(ideal) == n_samples`` and the left-hand side is an integer.
    # Floating-point rounding can only move the boundary by a negligible
    # amount, so ``missing`` is normally in ``[0, n_sources]``.
    missing = n_samples - int(quotas.sum())
    if missing == 0:
        return quotas

    # Only sources that actually carry weight may receive a row.  This is
    # what guarantees that a zero-weight source (e.g. a cluster whose class
    # points all coincide) is never rounded up.
    eligible = np.flatnonzero(weights > 0)
    remainders = ideal - quotas
    rng = check_random_state(random_state)
    tie_rank = np.empty(weights.size, dtype=np.int64)
    tie_rank[rng.permutation(weights.size)] = np.arange(weights.size)
    # Largest fractional part first; ties follow the seeded permutation so
    # that no source is systematically favoured by its position.
    order = eligible[np.lexsort((tie_rank[eligible], -remainders[eligible]))]

    if missing > 0:
        if missing <= order.size:
            quotas[order[:missing]] += 1
        else:  # pragma: no cover - only reachable through floating-point noise
            # The mathematical construction guarantees ``missing <
            # order.size``.  Should accumulated rounding ever violate it,
            # keep the total exact by cycling over the positive-weight
            # sources rather than touching a zero-weight one.
            quotas[order[np.arange(missing) % order.size]] += 1
    else:  # pragma: no cover - only reachable through floating-point noise
        # Symmetric rounding guard: take the surplus back from the sources
        # with the smallest fractional parts, never driving a count negative.
        surplus = -missing
        for index in order[::-1]:
            if surplus == 0:
                break
            if quotas[index] > 0:
                quotas[index] -= 1
                surplus -= 1
    return quotas


def neighbors_excluding_self(neighbor_indices, own_positions, n_neighbors):
    """Keep ``n_neighbors`` neighbours per row, dropping each row's own index.

    Nearest-neighbour searches that query a dataset against itself usually
    return the query point as its own first neighbour.  That is not reliable
    when several rows share the exact same feature vector: depending on the
    order in which ties are resolved, the query row may not appear at all, or
    may appear in another column, and a *different* row with identical
    features is a perfectly valid neighbour.  Rows are therefore identified
    by position rather than by feature values: only the entry equal to the
    query row's own position is removed, and identical rows remain usable as
    neighbours.  When the query row does not appear in its own neighbour list
    (possible with tied distances), all listed rows are other rows and the
    first ``n_neighbors`` of them are kept.

    Parameters
    ----------
    neighbor_indices : ndarray of shape (n_queries, n_returned)
        Positions returned by a nearest-neighbour search of a dataset against
        itself.

    own_positions : ndarray of shape (n_queries,)
        Position of each query row in the dataset that was searched.

    n_neighbors : int
        Number of neighbours to keep per query row.

    Returns
    -------
    selected : ndarray of shape (n_queries, n_neighbors)
        Neighbour positions with the query row's own position removed.  The
        original neighbour order is preserved otherwise.

    Raises
    ------
    ValueError
        If a row does not have ``n_neighbors`` neighbours other than itself
        among ``neighbor_indices``.
    """
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.intp)
    own_positions = np.asarray(own_positions, dtype=np.intp)
    n_neighbors = operator.index(n_neighbors)
    if n_neighbors < 0:
        raise ValueError("n_neighbors cannot be negative.")
    n_queries = own_positions.shape[0]
    if neighbor_indices.shape[0] != n_queries:
        raise ValueError(
            "neighbor_indices and own_positions must have the same number of "
            f"rows. Got {neighbor_indices.shape[0]} and {n_queries}."
        )
    if n_queries == 0:
        return neighbor_indices[:, :n_neighbors]
    if neighbor_indices.shape[1] < n_neighbors:
        raise ValueError(
            "neighbor_indices must contain at least n_neighbors columns. Got "
            f"{neighbor_indices.shape[1]} for n_neighbors={n_neighbors}."
        )

    not_self = neighbor_indices != own_positions[:, np.newaxis]
    # ``False`` sorts before ``True``: valid neighbours first, in their
    # original order, followed by the query row itself (if it was returned).
    order = np.argsort(~not_self, axis=1, kind="stable")
    columns = order[:, :n_neighbors]
    selected = np.take_along_axis(neighbor_indices, columns, axis=1)
    n_self_selected = np.sum(selected == own_positions[:, np.newaxis], axis=1)
    if np.any(n_self_selected):
        worst = int(np.flatnonzero(n_self_selected)[0])
        raise ValueError(
            "Not enough neighbours remain after removing each query row from "
            "its own neighbour list "
            f"(row {worst} of the search only has {n_neighbors} returned "
            "neighbours, itself included)."
        )
    return selected
