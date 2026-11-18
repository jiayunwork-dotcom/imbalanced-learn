"""Class to perform random under-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils.validation import _num_samples

from ...utils import Substitution, check_target_type
from ...utils._docstring import _random_state_docstring
from ...utils._group_sampling import _check_groups, _select_groups
from ...utils._sklearn_compat import validate_data
from ...utils._validation import _check_X
from ..base import BaseUnderSampler


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    replacement : bool, default=False
        Whether the sample is with or without replacement.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    groups_ : ndarray of shape (n_samples,)
        Group identifier of each input row, as validated. Only defined when
        ``groups`` is passed to :meth:`fit_resample`.

    group_ids_ : ndarray of shape (n_selected_groups,)
        Identifiers of the groups kept in the training data.

    group_codes_ : ndarray of shape (n_selected_groups,)
        Integer codes of the groups kept in the training data. The codes refer
        to the order of first appearance of the groups in ``groups``.

    n_groups_ : int
        Total number of groups seen in the input.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    NearMiss : Undersample using near-miss samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Whole groups can be under-sampled instead of individual rows by passing
    ``groups`` to :meth:`fit_resample`.  In that mode each group is kept or
    dropped as a whole, the target of every class is the number of rows of the
    smallest class, and each class stays represented.  Grouping requires
    ``sampling_strategy='auto'`` and ``replacement=False``.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "replacement": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(
        self, *, sampling_strategy="auto", random_state=None, replacement=False
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.replacement = replacement

    def fit_resample(self, X, y, *, groups=None):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        groups : array-like of shape (n_samples,), default=None
            Integer or non-empty string identifier of the group of each row.
            When provided, whole groups are selected or left out together.  The
            target of each class is the number of rows of the smallest class in
            the input.  ``None`` keeps the historical row-wise behaviour.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        if groups is None:
            return super().fit_resample(X, y)
        return super().fit_resample(X, y, groups=groups)

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = _check_X(X)
        X, y = validate_data(self, X=X, y=y, reset=True, skip_check_array=True)
        return X, y, binarize_y

    def _validate_group_parameters(self):
        """Check the parameters that only make sense in grouped mode."""
        if self.sampling_strategy != "auto":
            raise ValueError(
                "Grouped under-sampling only supports "
                "sampling_strategy='auto'; got "
                f"{self.sampling_strategy!r}."
            )
        if self.replacement:
            raise ValueError(
                "Grouped under-sampling does not support replacement=True."
            )

    def _fit_resample_groups(self, X, y, groups):
        """Under-sample by dropping whole groups."""
        info = _check_groups(groups, _num_samples(X))
        self._validate_group_parameters()

        self.groups_ = info.ids[info.codes]
        self.n_groups_ = info.n_groups

        selected, counts, _ = _select_groups(
            y, info.codes, info.n_groups, random_state=self.random_state
        )

        selected_codes = np.flatnonzero(selected)
        self.group_codes_ = selected_codes
        self.group_ids_ = info.ids[selected_codes]

        mask = np.isin(info.codes, selected_codes)
        idx_under = np.flatnonzero(mask)
        self.sample_indices_ = idx_under

        return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _fit_resample(self, X, y, groups=None):
        if groups is not None:
            return self._fit_resample_groups(X, y, groups)
        # A previous fit in grouped mode must not leak its attributes.
        for attribute in (
            "groups_",
            "group_ids_",
            "group_codes_",
            "n_groups_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)

        random_state = check_random_state(self.random_state)

        idx_under = np.empty((0,), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self.replacement,
                )
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (
                    idx_under,
                    np.flatnonzero(y == target_class)[index_target_class],
                ),
                axis=0,
            )

        self.sample_indices_ = idx_under

        return _safe_indexing(X, idx_under), _safe_indexing(y, idx_under)

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
            "_xfail_checks": {
                "check_complex_data": "Robust to this type of data.",
            },
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.sampler_tags.sample_indices = True
        return tags
