"""Weight-conserving accounting for resampling.

This module implements the bookkeeping used by ``fit_resample_weighted`` and
by ``Pipeline(weight_resampling="conserve")``.  The guiding principle is that
resampling must not create mass out of thin air: every output row is funded
by the original observations it derives from, and the total mass handed out
by an original observation plus the mass dropped from it by cleaning steps
equals its input ``sample_weight``.

Allocation strategy (design decision)
-------------------------------------
Each original observation splits its weight **equally over the output rows it
funds** (a "per-source uniform split"):

- a kept original row is funded only by itself;
- a duplicated row of :class:`~imblearn.over_sampling.RandomOverSampler` is
  funded only by the row it copies;
- a synthetic sample of :class:`~imblearn.over_sampling.SMOTE` is funded only
  by the two endpoint observations that generated it (one share from each);
- a row removed by :class:`~imblearn.under_sampling.EditedNearestNeighbours`
  hands its remaining mass to ``dropped_weight`` and it is not redistributed.

The alternative of letting every copy carry the full original weight would
inflate the total mass of the resampled class by the over-sampling factor,
which is exactly the behaviour this bookkeeping is meant to avoid.  See the
user guide for a discussion of the trade-offs.

The flow of mass between input rows and output rows is stored as sparse
coordinate records (:class:`WeightContributions`) so that the memory footprint
stays proportional to the number of non-zero transfers and never to the
product of the number of input and output rows.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
from scipy import sparse

__all__ = [
    "WeightContributions",
    "WeightedResamplingResult",
    "ResamplingAudit",
    "compose_weighted_results",
    "validate_sample_weight",
]

# Tolerances used to reconcile the books, as per the specification.
RTOL = 1e-10
ATOL = 1e-12


def validate_sample_weight(sample_weight, n_samples):
    """Validate per-observation weights for weight-conserving resampling.

    The weights must be a one-dimensional array of length ``n_samples`` with
    finite, non-negative values and a strictly positive total mass.

    Parameters
    ----------
    sample_weight : array-like of shape (n_samples,)
        The weights to validate. The caller's array is never modified.

    n_samples : int
        The expected number of rows.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,), dtype=float64
        A private copy of the validated weights.

    Raises
    ------
    ValueError
        If the weights are not one-dimensional, do not match ``n_samples``,
        contain non-finite or negative values, or sum to zero.
    """
    if sample_weight is None:
        raise ValueError(
            "sample_weight cannot be None when weight-conserving resampling "
            "is requested."
        )
    try:
        raw = np.asarray(sample_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sample_weight must be a one-dimensional array of finite, "
            f"non-negative numbers. Got {sample_weight!r} instead."
        ) from exc
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(
            "sample_weight must be a one-dimensional array of finite, "
            "non-negative numbers. Got complex values instead."
        )
    try:
        weight = np.array(sample_weight, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sample_weight must be a one-dimensional array of finite, "
            f"non-negative numbers. Got {sample_weight!r} instead."
        ) from exc
    if weight.ndim != 1:
        raise ValueError(
            "sample_weight must be one-dimensional. Got an array with "
            f"{weight.ndim} dimensions instead."
        )
    if weight.shape[0] != n_samples:
        raise ValueError(
            "sample_weight must have as many entries as there are rows in X. "
            f"Got {weight.shape[0]} weights for {n_samples} rows."
        )
    if not np.all(np.isfinite(weight)):
        raise ValueError("sample_weight must contain only finite values.")
    if np.any(weight < 0):
        raise ValueError("sample_weight cannot contain negative values.")
    if not weight.sum() > 0:
        raise ValueError("sample_weight must have a strictly positive total mass.")
    return weight


class WeightContributions:
    """Non-zero mass transfers from input rows to output rows.

    Each record ``(input_index[k], output_index[k], mass[k])`` states that the
    input row at position ``input_index[k]`` funds the output row at position
    ``output_index[k]`` with ``mass[k]`` units of weight. Only non-zero
    transfers are recorded. Positions are purely positional: two rows with
    identical feature values or duplicated pandas indices remain two distinct
    sources.

    The records are unordered. Use :meth:`to_scipy_sparse` to obtain a
    canonical sparse matrix view of the transfers.

    Parameters
    ----------
    input_index : array-like of shape (n_records,)
        Positional index of the funding input row for each record.

    output_index : array-like of shape (n_records,)
        Positional index of the funded output row for each record.

    mass : array-like of shape (n_records,)
        Amount of weight transferred by each record. Records with a zero mass
        are dropped.

    n_inputs : int
        Number of rows of the input dataset.

    n_outputs : int
        Number of rows of the output dataset.

    Attributes
    ----------
    input_index : ndarray of shape (n_records,), dtype=int64
        Positional index of the funding input row for each record.

    output_index : ndarray of shape (n_records,), dtype=int64
        Positional index of the funded output row for each record.

    mass : ndarray of shape (n_records,), dtype=float64
        Amount of weight transferred by each record.

    n_inputs : int
        Number of rows of the input dataset.

    n_outputs : int
        Number of rows of the output dataset.
    """

    def __init__(self, input_index, output_index, mass, *, n_inputs, n_outputs):
        input_index = np.asarray(input_index, dtype=np.int64)
        output_index = np.asarray(output_index, dtype=np.int64)
        mass = np.asarray(mass, dtype=np.float64)
        if not (input_index.ndim == output_index.ndim == mass.ndim == 1):
            raise ValueError("input_index, output_index and mass must be 1-D arrays.")
        if not (input_index.shape == output_index.shape == mass.shape):
            raise ValueError(
                "input_index, output_index and mass must have the same length."
            )
        n_inputs = int(n_inputs)
        n_outputs = int(n_outputs)
        if n_inputs < 0 or n_outputs < 0:
            raise ValueError("n_inputs and n_outputs must be non-negative.")
        if input_index.size:
            if input_index.min() < 0 or input_index.max() >= n_inputs:
                raise ValueError("input_index contains out-of-bound positions.")
            if output_index.min() < 0 or output_index.max() >= n_outputs:
                raise ValueError("output_index contains out-of-bound positions.")
        if np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise ValueError("mass must contain finite, non-negative values.")
        # only non-zero transfers are recorded
        keep = mass != 0
        self.input_index = input_index[keep]
        self.output_index = output_index[keep]
        self.mass = mass[keep]
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    @property
    def shape(self):
        """Shape of the implicit transfer matrix: ``(n_inputs, n_outputs)``."""
        return (self.n_inputs, self.n_outputs)

    def __len__(self):
        return self.mass.shape[0]

    def __iter__(self):
        return iter(zip(self.input_index, self.output_index, self.mass))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_records={len(self)}, "
            f"n_inputs={self.n_inputs}, n_outputs={self.n_outputs})"
        )

    def to_scipy_sparse(self):
        """Return the transfers as a CSR matrix of shape (n_inputs, n_outputs).

        Returns
        -------
        flow : scipy.sparse.csr_matrix
            Sparse matrix such that ``flow[i, j]`` is the mass transferred
            from input row ``i`` to output row ``j``. Duplicate transfers are
            summed and explicit zeros are removed.
        """
        flow = sparse.csr_matrix(
            (self.mass, (self.input_index, self.output_index)),
            shape=(self.n_inputs, self.n_outputs),
        )
        flow.sum_duplicates()
        flow.eliminate_zeros()
        return flow

    @classmethod
    def from_scipy_sparse(cls, flow):
        """Build contributions from a sparse transfer matrix.

        Parameters
        ----------
        flow : scipy.sparse matrix of shape (n_inputs, n_outputs)
            Sparse matrix such that ``flow[i, j]`` is the mass transferred
            from input row ``i`` to output row ``j``.

        Returns
        -------
        contributions : WeightContributions
            The equivalent non-zero transfer records.
        """
        flow = sparse.coo_matrix(flow)
        return cls(
            flow.row,
            flow.col,
            flow.data,
            n_inputs=flow.shape[0],
            n_outputs=flow.shape[1],
        )

    def input_sums(self):
        """Total mass handed out by each input row.

        Returns
        -------
        ndarray of shape (n_inputs,)
            Per-input-row sum of the outgoing transfers.
        """
        return np.bincount(self.input_index, weights=self.mass, minlength=self.n_inputs)

    def output_sums(self):
        """Total mass received by each output row.

        Returns
        -------
        ndarray of shape (n_outputs,)
            Per-output-row sum of the incoming transfers, i.e. the training
            weight of each output row.
        """
        return np.bincount(
            self.output_index, weights=self.mass, minlength=self.n_outputs
        )

    def lift_dropped(self, dropped_next):
        """Map per-output-row dropped mass back to the input rows.

        Parameters
        ----------
        dropped_next : ndarray of shape (n_outputs,)
            Mass dropped from each output row of this transfer by a
            subsequent cleaning stage.

        Returns
        -------
        ndarray of shape (n_inputs,)
            The dropped mass expressed per input row: each input row is
            charged the fraction of the next stage's dropped mass that
            corresponds to its own transfers.
        """
        dropped_next = np.asarray(dropped_next, dtype=np.float64)
        if dropped_next.shape != (self.n_outputs,):
            raise ValueError(
                "dropped_next must have one entry per output row. Got shape "
                f"{dropped_next.shape} for {self.n_outputs} output rows."
            )
        # fraction of each output row's mass that the next stage dropped
        output_weight = self.output_sums()
        dropped_fraction = np.zeros(self.n_outputs, dtype=np.float64)
        np.divide(
            dropped_next,
            output_weight,
            out=dropped_fraction,
            where=output_weight > 0,
        )
        return np.bincount(
            self.input_index,
            weights=self.mass * dropped_fraction[self.output_index],
            minlength=self.n_inputs,
        )

    def compose(self, other):
        """Compose with the transfers of a subsequent resampling stage.

        Parameters
        ----------
        other : WeightContributions
            Transfers whose input rows are the output rows of ``self``, built
            with the weights ``self.output_sums()``.

        Returns
        -------
        WeightContributions
            Transfers from the input rows of ``self`` to the output rows of
            ``other``. Each transfer of ``other`` is weighted by the fraction
            of the intermediate row's mass it represents. The composition is
            a sparse matrix product; its cost and memory stay proportional to
            the number of non-zero transfers.
        """
        if not isinstance(other, WeightContributions):
            raise TypeError(
                "other must be a WeightContributions instance. Got "
                f"{type(other).__name__} instead."
            )
        if self.n_outputs != other.n_inputs:
            raise ValueError(
                "Cannot compose contributions: the first stage has "
                f"{self.n_outputs} output rows but the second stage expects "
                f"{other.n_inputs} input rows."
            )
        # transition matrix of the next stage: which fraction of each
        # intermediate row's mass lands on each final row
        intermediate_weight = self.output_sums()
        inverse_weight = np.zeros(self.n_outputs, dtype=np.float64)
        np.divide(
            1.0, intermediate_weight, out=inverse_weight, where=intermediate_weight > 0
        )
        transition = sparse.diags(inverse_weight) @ other.to_scipy_sparse()
        return WeightContributions.from_scipy_sparse(
            self.to_scipy_sparse() @ transition
        )


class WeightedResamplingResult:
    """Result of :meth:`fit_resample_weighted` of a sampler.

    Attributes
    ----------
    X : {ndarray, dataframe, sparse matrix} of shape (n_samples_new, n_features)
        The resampled data. Identical to the output of ``fit_resample`` with
        the same parameters and random state.

    y : {ndarray, series} of shape (n_samples_new,)
        The resampled target. Identical to the output of ``fit_resample``
        with the same parameters and random state.

    sample_weight : ndarray of shape (n_samples_new,)
        The training weight of each output row, aligned with ``X`` and ``y``.

    contributions : WeightContributions
        The non-zero ``(input position, output position, mass)`` transfers
        from the rows of the original input to the rows of ``X``.

    dropped_weight : ndarray of shape (n_samples,)
        The mass dropped from each row of the original input, e.g. by a
        cleaning sampler. Over-samplers never drop mass.
    """

    def __init__(self, *, X, y, sample_weight, contributions, dropped_weight):
        if not isinstance(contributions, WeightContributions):
            raise TypeError(
                "contributions must be a WeightContributions instance. Got "
                f"{type(contributions).__name__} instead."
            )
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        dropped_weight = np.asarray(dropped_weight, dtype=np.float64)
        if sample_weight.shape != (contributions.n_outputs,):
            raise ValueError(
                "sample_weight must have one entry per output row. Got shape "
                f"{sample_weight.shape} for {contributions.n_outputs} output rows."
            )
        if dropped_weight.shape != (contributions.n_inputs,):
            raise ValueError(
                "dropped_weight must have one entry per input row. Got shape "
                f"{dropped_weight.shape} for {contributions.n_inputs} input rows."
            )
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.contributions = contributions
        self.dropped_weight = dropped_weight

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_inputs={self.contributions.n_inputs}, "
            f"n_outputs={self.contributions.n_outputs}, "
            f"mass_out={self.sample_weight.sum():.6g}, "
            f"mass_dropped={self.dropped_weight.sum():.6g})"
        )

    def is_conserved(self, sample_weight_in, *, rtol=RTOL, atol=ATOL):
        """Check that the books balance for the given input weights.

        Parameters
        ----------
        sample_weight_in : array-like of shape (n_samples,)
            The weights handed to ``fit_resample_weighted``.

        rtol : float, default=1e-10
            Relative tolerance of the comparison.

        atol : float, default=1e-12
            Absolute tolerance of the comparison.

        Returns
        -------
        bool
            True when, for every input row, the mass it handed to output rows
            plus its dropped mass equals its input weight, and, for every
            output row, the mass it received equals its training weight.
        """
        sample_weight_in = np.asarray(sample_weight_in, dtype=np.float64)
        input_balance = self.contributions.input_sums() + self.dropped_weight
        return bool(
            np.allclose(input_balance, sample_weight_in, rtol=rtol, atol=atol)
            and np.allclose(
                self.sample_weight,
                self.contributions.output_sums(),
                rtol=rtol,
                atol=atol,
            )
        )

    def check_conserved(self, sample_weight_in, *, rtol=RTOL, atol=ATOL):
        """Raise an ``AssertionError`` if the books do not balance.

        Parameters
        ----------
        sample_weight_in : array-like of shape (n_samples,)
            The weights handed to ``fit_resample_weighted``.

        rtol : float, default=1e-10
            Relative tolerance of the comparison.

        atol : float, default=1e-12
            Absolute tolerance of the comparison.
        """
        sample_weight_in = np.asarray(sample_weight_in, dtype=np.float64)
        input_balance = self.contributions.input_sums() + self.dropped_weight
        if not np.allclose(input_balance, sample_weight_in, rtol=rtol, atol=atol):
            worst = int(np.argmax(np.abs(input_balance - sample_weight_in)))
            raise AssertionError(
                "Weight conservation is violated for the input rows: input "
                f"row {worst} handed out {input_balance[worst]!r} (transfers "
                "plus dropped mass) but weighs "
                f"{sample_weight_in[worst]!r}."
            )
        output_sums = self.contributions.output_sums()
        if not np.allclose(self.sample_weight, output_sums, rtol=rtol, atol=atol):
            worst = int(np.argmax(np.abs(self.sample_weight - output_sums)))
            raise AssertionError(
                "Weight conservation is violated for the output rows: output "
                f"row {worst} received {output_sums[worst]!r} but its "
                f"training weight is {self.sample_weight[worst]!r}."
            )


class ResamplingAudit:
    """Weight-conservation audit left on a fitted pipeline.

    This is the reconciliation left by
    ``Pipeline.fit(X, y, sample_weight=...)`` when
    ``weight_resampling="conserve"`` is enabled. All positions are rooted at
    the rows of the original ``X`` passed to ``Pipeline.fit``, regardless of
    how many samplers were chained.

    Attributes
    ----------
    sample_weight_in : ndarray of shape (n_samples,)
        The validated weights of the rows passed to ``Pipeline.fit``.

    sample_weight_out : ndarray of shape (n_samples_new,)
        The weights handed to the final estimator, aligned with the final
        training data.

    contributions : WeightContributions
        The non-zero ``(original position, final position, mass)`` transfers
        from the rows passed to ``Pipeline.fit`` to the rows of the final
        training data.

    dropped_weight : ndarray of shape (n_samples,)
        The mass dropped from each original row by cleaning steps.
    """

    def __init__(
        self, *, sample_weight_in, sample_weight_out, contributions, dropped_weight
    ):
        if not isinstance(contributions, WeightContributions):
            raise TypeError(
                "contributions must be a WeightContributions instance. Got "
                f"{type(contributions).__name__} instead."
            )
        self.sample_weight_in = np.asarray(sample_weight_in, dtype=np.float64)
        self.sample_weight_out = np.asarray(sample_weight_out, dtype=np.float64)
        self.contributions = contributions
        self.dropped_weight = np.asarray(dropped_weight, dtype=np.float64)
        if self.sample_weight_in.shape != (contributions.n_inputs,):
            raise ValueError("sample_weight_in must have one entry per original row.")
        if self.sample_weight_out.shape != (contributions.n_outputs,):
            raise ValueError("sample_weight_out must have one entry per final row.")
        if self.dropped_weight.shape != (contributions.n_inputs,):
            raise ValueError("dropped_weight must have one entry per original row.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_inputs={self.contributions.n_inputs}, "
            f"n_outputs={self.contributions.n_outputs}, "
            f"mass_in={self.sample_weight_in.sum():.6g}, "
            f"mass_out={self.sample_weight_out.sum():.6g}, "
            f"mass_dropped={self.dropped_weight.sum():.6g})"
        )

    def is_conserved(self, *, rtol=RTOL, atol=ATOL):
        """Check that the books balance.

        Parameters
        ----------
        rtol : float, default=1e-10
            Relative tolerance of the comparison.

        atol : float, default=1e-12
            Absolute tolerance of the comparison.

        Returns
        -------
        bool
            True when, for every original row, the mass it handed to final
            rows plus its dropped mass equals its input weight, and, for
            every final row, the mass it received equals the weight handed to
            the final estimator.
        """
        input_balance = self.contributions.input_sums() + self.dropped_weight
        return bool(
            np.allclose(input_balance, self.sample_weight_in, rtol=rtol, atol=atol)
            and np.allclose(
                self.sample_weight_out,
                self.contributions.output_sums(),
                rtol=rtol,
                atol=atol,
            )
        )

    def check_conserved(self, *, rtol=RTOL, atol=ATOL):
        """Raise an ``AssertionError`` if the books do not balance.

        Parameters
        ----------
        rtol : float, default=1e-10
            Relative tolerance of the comparison.

        atol : float, default=1e-12
            Absolute tolerance of the comparison.
        """
        input_balance = self.contributions.input_sums() + self.dropped_weight
        if not np.allclose(input_balance, self.sample_weight_in, rtol=rtol, atol=atol):
            worst = int(np.argmax(np.abs(input_balance - self.sample_weight_in)))
            raise AssertionError(
                "Weight conservation is violated for the original rows: row "
                f"{worst} handed out {input_balance[worst]!r} (transfers plus "
                f"dropped mass) but weighs {self.sample_weight_in[worst]!r}."
            )
        output_sums = self.contributions.output_sums()
        if not np.allclose(self.sample_weight_out, output_sums, rtol=rtol, atol=atol):
            worst = int(np.argmax(np.abs(self.sample_weight_out - output_sums)))
            raise AssertionError(
                "Weight conservation is violated for the final rows: row "
                f"{worst} received {output_sums[worst]!r} but the final "
                f"estimator was given {self.sample_weight_out[worst]!r}."
            )


def compose_weighted_results(first, second):
    """Compose the accounting of two sequential weighted resamplings.

    Parameters
    ----------
    first : WeightedResamplingResult
        Result of the first resampling stage.

    second : WeightedResamplingResult
        Result of the second resampling stage, applied to the output of
        ``first``.

    Returns
    -------
    WeightedResamplingResult
        The resampled data of ``second`` with the accounting rooted at the
        input rows of ``first``.
    """
    if not isinstance(first, WeightedResamplingResult) or not isinstance(
        second, WeightedResamplingResult
    ):
        raise TypeError("first and second must be WeightedResamplingResult instances.")
    contributions = first.contributions.compose(second.contributions)
    dropped_weight = first.dropped_weight + first.contributions.lift_dropped(
        second.dropped_weight
    )
    return WeightedResamplingResult(
        X=second.X,
        y=second.y,
        sample_weight=second.sample_weight,
        contributions=contributions,
        dropped_weight=dropped_weight,
    )
