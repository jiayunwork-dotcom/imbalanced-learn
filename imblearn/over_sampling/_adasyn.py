"""Class to perform over-sampling using ADASYN."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers

import numpy as np
from scipy import sparse
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils._param_validation import HasMethods, Interval

from ..utils import Substitution, check_neighbors_object
from ..utils._docstring import _random_state_docstring
from ._planning import neighbors_excluding_self
from .base import BaseOverSampler


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class ADASYN(BaseOverSampler):
    """Oversample using Adaptive Synthetic (ADASYN) algorithm.

    This method is similar to SMOTE but it generates different number of
    samples depending on an estimate of the local distribution of the class
    to be oversampled.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    n_neighbors : int or estimator object, default=5
        The nearest neighbors used to define the neighborhood of samples to use
        to generate the synthetic samples. You can pass:

        - an `int` corresponding to the number of neighbors to use. A
          `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this
          case.
        - an instance of a compatible nearest neighbors algorithm that should
          implement both methods `kneighbors` and `kneighbors_graph`. For
          instance, it could correspond to a
          :class:`~sklearn.neighbors.NearestNeighbors` but could be extended to
          any compatible class.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_ : estimator object
        Validated K-nearest Neighbours estimator linked to the parameter `n_neighbors`.

    _source_quotas_ : dict
        For each class actually over-sampled, the integer number of rows
        allocated to each anchor, in the ascending order of the class rows in
        ``X``. The counts sum exactly to the number of additional rows parsed
        from ``sampling_strategy_``. Exposed for inspection; not part of the
        stability guarantees.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

    Notes
    -----
    The implementation is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used.

    The difficulty of an anchor is estimated over the whole data set as the
    fraction of its nearest neighbours that belong to another class. The
    number of rows to generate for a class is the number of additional rows
    requested through ``sampling_strategy_``; it is split over the anchors by
    the shared largest-remainder rule (see
    :mod:`imblearn.over_sampling._planning`), so the total is exact and no
    generated row is discarded afterwards. Anchors whose ratio is zero are
    never allocated a row. The interpolation neighbours are then looked up
    only among the samples of the class being over-sampled, and only for the
    anchors that received a row. Rows are identified by position, so a row
    with the same features as the anchor is still a valid neighbour while the
    anchor's own row is not. If every anchor has a zero ratio, a
    ``RuntimeError`` is raised because the data set is not suited for ADASYN.

    References
    ----------
    .. [1] He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. "ADASYN:
       Adaptive synthetic sampling approach for imbalanced learning," In IEEE
       International Joint Conference on Neural Networks (IEEE World Congress
       on Computational Intelligence), pp. 1322-1328, 2008.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import ADASYN
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000,
    ... random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ada = ADASYN(random_state=42)
    >>> X_res, y_res = ada.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "n_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        n_neighbors=5,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def _validate_estimator(self):
        """Create the necessary objects for ADASYN"""
        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1
        )

    def _fit_resample(self, X, y):
        self._validate_estimator()
        random_state = check_random_state(self.random_state)
        # Per-class integer plan, aligned with the ascending positions of the
        # class rows in ``X``.  Recorded so that callers and tests can check
        # that the generation honours the shared allocation rule.
        self._source_quotas_ = {}

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)
            if X_class.shape[0] < 2:
                raise ValueError(
                    f"Class {class_sample} contains a single sample: ADASYN "
                    "needs at least two samples of the class to estimate the "
                    "difficulty of an anchor and to interpolate from."
                )

            self.nn_.fit(X)
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.  At least one neighbour
            # other than the anchor itself is required, and the search asks
            # for as many neighbours as available rather than failing when
            # the dataset holds only one or two fewer rows than requested.
            k_total = max(self.nn_.n_neighbors, 2)
            k_nn = min(k_total, X.shape[0])
            n_neighbors = k_nn - 1
            nns_all = self.nn_.kneighbors(
                X_class, n_neighbors=k_nn, return_distance=False
            )
            # Drop each anchor's own row by position: other rows with
            # identical features remain valid neighbours.
            nns_all = neighbors_excluding_self(
                nns_all,
                target_class_indices,
                min(n_neighbors, X.shape[0] - 1),
            )
            # Each anchor's difficulty is the fraction of its neighbours that
            # belong to another class.  An anchor whose neighbourhood contains
            # no other class never contributes, whatever the requested total.
            ratio_nn = np.mean(y[nns_all] != class_sample, axis=1)
            if not np.sum(ratio_nn):
                raise RuntimeError(
                    "Not any neigbours belong to the majority"
                    " class. This case will induce a NaN case"
                    " with a division by zero. ADASYN is not"
                    " suited for this specific dataset."
                    " Use SMOTE instead."
                )
            # Exact integer number of rows per anchor.  The plan sums to
            # ``n_samples`` by construction: anchors are never truncated
            # afterwards and rows are never missing because rounding was
            # applied anchor by anchor.
            n_samples_generate = self._allocate_source_quotas(ratio_nn, n_samples)
            self._source_quotas_[class_sample] = n_samples_generate
            allocated = np.flatnonzero(n_samples_generate > 0)

            # The same-class neighbours used for the interpolation are only
            # looked up for the anchors that actually received a quota.  The
            # search is restricted to the samples of the current class, so
            # every selected neighbour belongs to that class.
            self.nn_.fit(X_class)
            k_class = min(k_nn, X_class.shape[0])
            nns = self.nn_.kneighbors(
                _safe_indexing(X_class, allocated),
                n_neighbors=k_class,
                return_distance=False,
            )
            nns = neighbors_excluding_self(
                nns, allocated, min(n_neighbors, X_class.shape[0] - 1)
            )
            if nns.shape[1] == 0:
                raise ValueError(
                    f"Class {class_sample} contains a single sample: ADASYN "
                    "needs another sample of the same class to interpolate "
                    "from."
                )

            per_anchor = n_samples_generate[allocated]
            rows = np.repeat(np.arange(allocated.size), per_anchor)
            cols = random_state.randint(0, nns.shape[1], size=rows.size)
            base = _safe_indexing(X_class, allocated[rows])
            diffs = _safe_indexing(X_class, nns[rows, cols]) - base
            steps = random_state.uniform(size=(rows.size, 1))

            if sparse.issparse(X):
                sparse_func = type(X).__name__
                steps = getattr(sparse, sparse_func)(steps)
                X_new = base + steps.multiply(diffs)
            else:
                X_new = base + steps * diffs

            X_new = X_new.astype(X.dtype)
            y_new = np.full(rows.size, fill_value=class_sample, dtype=y.dtype)
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
