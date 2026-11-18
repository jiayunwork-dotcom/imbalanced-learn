"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils._metadata_requests import METHODS
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _num_samples

from .utils import check_sampling_strategy, check_target_type
from .utils._sklearn_compat import _fit_context, get_tags, validate_data
from .utils._validation import ArraysTransformer
from .utils._weight_accounting import WeightedResamplingResult, validate_sample_weight

if "fit_predict" not in METHODS:
    METHODS.append("fit_predict")
if "fit_transform" not in METHODS:
    METHODS.append("fit_transform")
METHODS.append("fit_resample")

try:
    from sklearn.utils._metadata_requests import SIMPLE_METHODS

    SIMPLE_METHODS.append("fit_resample")
except ImportError:
    # in older versions of scikit-learn, only METHODS is used
    pass


class SamplerMixin(metaclass=ABCMeta):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = "sampler"

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **params):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_resample(self, X, y, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        output = self._fit_resample(X, y, **params)

        y_ = (
            label_binarize(output[1], classes=np.unique(y)) if binarize_y else output[1]
        )

        X_, y_ = arrays_transformer.transform(output[0], y_)
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    @abstractmethod
    def _fit_resample(self, X, y, **params):
        """Base method defined in each sampler to defined the sampling
        strategy.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        """
        pass


class BaseSampler(SamplerMixin, OneToOneFeatureMixin, BaseEstimator):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = validate_data(self, X=X, y=y, reset=True, accept_sparse=accept_sparse)
        return X, y, binarize_y

    def fit(self, X, y, **params):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        return super().fit(X, y, **params)

    def fit_resample(self, X, y, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        return super().fit_resample(X, y, **params)

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse", "dataframe"]}

    def __sklearn_tags__(self):
        from .utils._sklearn_compat import TargetTags
        from .utils._tags import InputTags, SamplerTags, Tags

        tags = Tags(
            estimator_type="sampler",
            target_tags=TargetTags(required=True),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
            sampler_tags=SamplerTags(),
        )
        tags.input_tags = InputTags()
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = True
        tags.input_tags.dataframe = True
        return tags


def _identity(X, y):
    return X, y


def _supports_weighted_resampling(estimator):
    """Return True if `estimator` supports weight-conserving resampling.

    The capability is declared per concrete class with the class attribute
    ``_weighted_resampling_supported = True`` and is deliberately *not*
    inherited: subclasses implementing a different resampling logic (e.g.
    :class:`~imblearn.over_sampling.SMOTENC`) would record an invalid
    provenance if they reused the bookkeeping of their parent class.
    """
    return bool(type(estimator).__dict__.get("_weighted_resampling_supported", False))


def _fit_resample_weighted(sampler, X, y, sample_weight):
    """Shared scaffolding of the ``fit_resample_weighted`` methods.

    This mirrors :meth:`SamplerMixin.fit_resample` and additionally builds
    the weight-conservation bookkeeping. The actual sampling is delegated to
    ``sampler._fit_resample_flow`` which must produce the exact same ``X``
    and ``y`` as ``sampler._fit_resample``.
    """
    if not _supports_weighted_resampling(sampler):
        raise NotImplementedError(
            f"{type(sampler).__name__} does not support weight-conserving "
            "resampling. Only RandomOverSampler (with shrinkage=None), SMOTE, "
            "EditedNearestNeighbours and SMOTEENN implement "
            "fit_resample_weighted."
        )
    sample_weight = validate_sample_weight(sample_weight, _num_samples(X))
    check_classification_targets(y)
    arrays_transformer = ArraysTransformer(X, y)
    X, y, binarize_y = sampler._check_X_y(X, y)

    sampler.sampling_strategy_ = check_sampling_strategy(
        sampler.sampling_strategy, y, sampler._sampling_type
    )

    X_res, y_res, contributions, dropped_weight = sampler._fit_resample_flow(
        X, y, sample_weight
    )

    y_ = label_binarize(y_res, classes=np.unique(y)) if binarize_y else y_res

    X_, y_ = arrays_transformer.transform(X_res, y_)
    return WeightedResamplingResult(
        X=X_,
        y=y_,
        sample_weight=contributions.output_sums(),
        contributions=contributions,
        dropped_weight=dropped_weight,
    )


_FIT_RESAMPLE_WEIGHTED_DOCSTRING = """Resample the dataset and track each weight.

    This is the weight-conserving counterpart of ``fit_resample``: the
    resampled data are identical to what ``fit_resample`` returns with the
    same parameters and random state, and each output row is additionally
    given a training weight funded by the original observations it derives
    from. The total mass is conserved: the weight of an original observation
    is split over the output rows it funds and, for rows removed by a
    cleaning step, recorded as dropped mass.

    Read more in the :ref:`User Guide <weight_conservation>`.

    Parameters
    ----------
    X : {array-like, dataframe, sparse matrix} of shape \\
            (n_samples, n_features)
        Matrix containing the data which have to be sampled.

    y : array-like of shape (n_samples,)
        Corresponding label for each sample in X.

    sample_weight : array-like of shape (n_samples,)
        The weight of each original observation. It must be one-dimensional,
        of the same length as ``X``, finite, non-negative and with a strictly
        positive total mass, otherwise a ``ValueError`` is raised and no
        resampling is performed. The array is copied and never modified.

    Returns
    -------
    result : WeightedResamplingResult
        The resampled ``X``, ``y`` and their ``sample_weight`` aligned with
        the output rows, the non-zero ``contributions`` records
        ``(input position, output position, mass)`` and the
        ``dropped_weight`` of each input row.

    Notes
    -----
    Each original observation splits its weight equally over the output rows
    it funds: its own kept row, the rows copied from it, and the synthetic
    samples for which it was one of the two generating endpoints. A synthetic
    sample is funded by its two endpoints only, a copied row by its source
    row only, and a kept original row by itself only.
    """


def _document_fit_resample_weighted(func):
    """Attach the shared ``fit_resample_weighted`` docstring."""
    func.__doc__ = _FIT_RESAMPLE_WEIGHTED_DOCSTRING
    return func


def is_sampler(estimator):
    """Return True if the given estimator is a sampler, False otherwise.

    Parameters
    ----------
    estimator : object
        Estimator to test.

    Returns
    -------
    is_sampler : bool
        True if estimator is a sampler, otherwise False.
    """

    if hasattr(estimator, "_estimator_type") and estimator._estimator_type == "sampler":
        return True
    tags = get_tags(estimator)
    if hasattr(tags, "sampler_tags") and tags.sampler_tags is not None:
        return True
    return False


class FunctionSampler(BaseSampler):
    """Construct a sampler from calling an arbitrary callable.

    Read more in the :ref:`User Guide <function_sampler>`.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed the
        same arguments as transform, with args and kwargs forwarded. If func is
        None, then func will be the identity function.

    accept_sparse : bool, default=True
        Whether sparse input are supported. By default, sparse inputs are
        supported.

    kw_args : dict, default=None
        The keyword argument expected by ``func``.

    validate : bool, default=True
        Whether or not to bypass the validation of ``X`` and ``y``. Turning-off
        validation allows to use the ``FunctionSampler`` with any type of
        data.

        .. versionadded:: 0.6

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    sklearn.preprocessing.FunctionTransfomer : Stateless transformer.

    Notes
    -----
    See
    :ref:`sphx_glr_auto_examples_applications_plot_outlier_rejections.py`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from imblearn import FunctionSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

    We can create to select only the first ten samples for instance.

    >>> def func(X, y):
    ...   return X[:10], y[:10]
    >>> sampler = FunctionSampler(func=func)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> np.all(X_res == X[:10])
    True
    >>> np.all(y_res == y[:10])
    True

    We can also create a specific function which take some arguments.

    >>> from collections import Counter
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> def func(X, y, sampling_strategy, random_state):
    ...   return RandomUnderSampler(
    ...       sampling_strategy=sampling_strategy,
    ...       random_state=random_state).fit_resample(X, y)
    >>> sampler = FunctionSampler(func=func,
    ...                           kw_args={'sampling_strategy': 'auto',
    ...                                    'random_state': 0})
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print(f'Resampled dataset shape {sorted(Counter(y_res).items())}')
    Resampled dataset shape [(0, 100), (1, 100)]
    """

    _sampling_type = "bypass"

    _parameter_constraints: dict = {
        "func": [callable, None],
        "accept_sparse": ["boolean"],
        "kw_args": [dict, None],
        "validate": ["boolean"],
    }

    def __init__(self, *, func=None, accept_sparse=True, kw_args=None, validate=True):
        super().__init__()
        self.func = func
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args
        self.validate = validate

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        self._validate_params()
        # we need to overwrite SamplerMixin.fit to bypass the validation
        if self.validate:
            check_classification_targets(y)
            X, y, _ = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        return self

    def fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        self._validate_params()
        arrays_transformer = ArraysTransformer(X, y)

        if self.validate:
            check_classification_targets(y)
            X, y, binarize_y = self._check_X_y(X, y, accept_sparse=self.accept_sparse)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        output = self._fit_resample(X, y)

        if self.validate:
            y_ = (
                label_binarize(output[1], classes=np.unique(y))
                if binarize_y
                else output[1]
            )
            X_, y_ = arrays_transformer.transform(output[0], y_)
            return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

        return output

    def _fit_resample(self, X, y):
        func = _identity if self.func is None else self.func
        output = func(X, y, **(self.kw_args if self.kw_args else {}))
        return output
