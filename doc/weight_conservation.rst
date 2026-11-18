.. _weight_conservation:

================================
Weight-conserving resampling
================================

.. currentmodule:: imblearn

Resampling changes the *number* of rows of a dataset. When the rows carry
weights, duplicating or synthesizing rows should not conjure new mass out of
thin air: the total weight that a set of original observations represents
stays the same, it is only redistributed over more rows. The
``fit_resample_weighted`` methods and the ``weight_resampling='conserve'``
pipeline mode implement exactly this bookkeeping.

.. _weight_conservation_samplers:

Weighted resampling of a single sampler
---------------------------------------

The following samplers provide a ``fit_resample_weighted`` method:

- :class:`~imblearn.over_sampling.RandomOverSampler` (with
  ``shrinkage=None``);
- :class:`~imblearn.over_sampling.SMOTE`;
- :class:`~imblearn.under_sampling.EditedNearestNeighbours`;
- :class:`~imblearn.combine.SMOTEENN`.

The method mirrors ``fit_resample`` -- the resampled ``X`` and ``y`` are
identical for the same parameters and random state -- and additionally
returns the weight bookkeeping::

  >>> from collections import Counter
  >>> import numpy as np
  >>> from sklearn.datasets import make_classification
  >>> from imblearn.over_sampling import SMOTE
  >>> X, y = make_classification(
  ...     n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3,
  ...     n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1,
  ...     n_samples=1000, random_state=10)
  >>> sample_weight = np.random.RandomState(0).rand(X.shape[0]) + 0.1
  >>> result = SMOTE(random_state=42).fit_resample_weighted(
  ...     X, y, sample_weight=sample_weight)

The result exposes:

- ``X``, ``y`` and ``sample_weight``, aligned with the output rows;
- ``contributions``, the non-zero ``(input position, output position,
  mass)`` records describing which original observation funds which output
  row;
- ``dropped_weight``, the mass dropped from each input row (only cleaning
  samplers such as
  :class:`~imblearn.under_sampling.EditedNearestNeighbours` drop mass).

The books balance per source: the mass an input row hands to the output rows
plus its dropped mass equals its input weight, and the mass received by an
output row equals its training weight. This can be verified with a relative
tolerance of ``1e-10`` and an absolute tolerance of ``1e-12``::

  >>> result.check_conserved(sample_weight)  # raises if the books do not balance
  >>> result.is_conserved(sample_weight)
  True

The input weights must be one-dimensional, of the same length as ``X``,
finite, non-negative and with a strictly positive total mass, otherwise a
:class:`ValueError` is raised before any resampling happens. The caller's
arrays are never modified.

.. _weight_conservation_strategy:

How original and synthetic rows share the mass
-----------------------------------------------

Each original observation splits its weight **equally over the output rows it
funds**:

- a kept original row is funded only by itself;
- a row copied by :class:`~imblearn.over_sampling.RandomOverSampler` is
  funded only by the observation it duplicates: an observation copied ``k``
  times spends ``w / k`` on each copy;
- a synthetic sample of :class:`~imblearn.over_sampling.SMOTE` is funded only
  by the two endpoint observations that generated it and receives one share
  from each: an observation that is the endpoint of ``t`` synthetic samples
  keeps ``w / (1 + t)`` for its own row and spends the same share on each of
  those synthetic samples;
- a row removed by :class:`~imblearn.under_sampling.EditedNearestNeighbours`
  hands its whole weight to ``dropped_weight``; it is neither redistributed
  to the survivors nor renormalized.

With this strategy, synthetic samples genuinely participate in training
(they carry a positive weight whenever one of their endpoints does) while
the total mass of each original observation -- and therefore of each class --
is exactly conserved. The geometric coverage of the minority class improves
because its mass is spread over more distinct locations of the feature
space, but the class does not become "more important" globally: the sum of
the weights of a class after resampling equals its sum before.

An alternative strategy would let every copy (and synthetic row) carry the
*full* original weight, inflating the total mass by the over-sampling
factor. This preserves the per-row meaning of the weights, but it multiplies
the total evidence of the minority class -- exactly what weight conservation
is meant to avoid. The training effect differs accordingly: with mass
inflation, a class over-sampled by a factor ``k`` shifts the decision
boundary as if it were ``k`` times more prevalent, and its effective sample
size (Kish's ``(sum w)^2 / sum w^2``) grows as if ``k`` times more
independent evidence had been observed, overstating the confidence of the
fit. With the conserving split, the class keeps its original total mass and
the benefit comes only from a better geometric coverage per unit of mass.

.. _weight_conservation_composition:

Chaining several samplers
-------------------------

When several samplers are chained, the transfers compose: the mass of an
original observation follows the rows it funded through each stage, and the
mass dropped by a cleaning stage is charged to the original observations in
proportion to their transfers. The composition never materializes a dense
``n_input_rows x n_output_rows`` matrix: the transfers are stored as sparse
coordinate records and composed as sparse matrices, so the memory footprint
stays proportional to the number of non-zero transfers (at most one record
per kept row and per copy, and two per synthetic sample).

:func:`~imblearn.utils.compose_weighted_results` combines the bookkeeping of
two sequential weighted resamplings::

  >>> from imblearn.under_sampling import EditedNearestNeighbours
  >>> from imblearn.utils import compose_weighted_results
  >>> first = SMOTE(random_state=42).fit_resample_weighted(
  ...     X, y, sample_weight=sample_weight)
  >>> second = EditedNearestNeighbours(
  ...     sampling_strategy="all").fit_resample_weighted(
  ...     first.X, first.y, first.sample_weight)
  >>> combined = compose_weighted_results(first, second)
  >>> combined.check_conserved(sample_weight)

This is exactly what :class:`~imblearn.combine.SMOTEENN` computes in a
single call, with the same parameters and random state::

  >>> from imblearn.combine import SMOTEENN
  >>> combo = SMOTEENN(random_state=42).fit_resample_weighted(
  ...     X, y, sample_weight=sample_weight)
  >>> np.allclose(combo.sample_weight, combined.sample_weight)
  True

.. _weight_conservation_pipeline:

Weight-conserving pipelines
---------------------------

Passing ``weight_resampling='conserve'`` to
:class:`~imblearn.pipeline.Pipeline` enables the conservation machinery in
:meth:`~imblearn.pipeline.Pipeline.fit`. The ``sample_weight`` argument of
``fit`` is then reserved for the conserved weights: it is validated, carried
through the samplers of the pipeline, and the weights of the final training
rows are handed to the final estimator::

  >>> from sklearn.linear_model import LogisticRegression
  >>> from imblearn.pipeline import Pipeline
  >>> pipe = Pipeline(
  ...     [("smote", SMOTE(random_state=42)),
  ...      ("clf", LogisticRegression(max_iter=500))],
  ...     weight_resampling="conserve")
  >>> pipe.fit(X, y, sample_weight=sample_weight)
  Pipeline(steps=[('smote', SMOTE(random_state=42)),
                  ('clf', LogisticRegression(max_iter=500))],
           weight_resampling='conserve')

The reconciliation of the fit is stored in ``resampling_audit_``, rooted at
the rows passed to ``fit`` no matter how many samplers were chained::

  >>> pipe.resampling_audit_.is_conserved()
  True

Transformers that preserve the row order (e.g.
:class:`~sklearn.preprocessing.StandardScaler`, ``'passthrough'``) can be
interleaved with the samplers. Every sampler in the pipeline must implement
``fit_resample_weighted`` and the final estimator must accept a
``sample_weight`` parameter in its ``fit`` method, otherwise an error is
raised before any fitting happens. If no ``sample_weight`` is given, uniform
weights are used. Other metadata keep their usual routing rules, with or
without metadata routing enabled, and ``predict``/``score`` never resample:
the ``sample_weight`` of ``score`` refers to the test rows, as usual.

The caching of the pipeline (``memory``) is aware of the weights: the same
data with different weights never hits a stale entry, and a warm cache
returns the same books as a cold one.
