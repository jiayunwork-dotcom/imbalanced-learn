"""Check the weight accounting on a larger CSR dataset.

The accounting must stay cheap: the transfers are stored as sparse coordinate
records and composed as sparse matrices, so the extra peak memory of
``fit_resample_weighted`` compared to ``fit_resample`` with the same
parameters must remain small. The peak memory is measured in subprocesses
with the resident set size high-water mark; both variants perform the exact
same nearest-neighbour search (same data, same seed), so the difference
isolates the cost of the bookkeeping.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification

from imblearn.over_sampling import SMOTE

RTOL = 1e-10
ATOL = 1e-12

N_SAMPLES = 20_000
MAX_OUTPUT_ROWS = 40_000
MEMORY_BUDGET_MIB = 128


def _make_sparse_data():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=30,
        n_informative=6,
        n_redundant=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        class_sep=1.5,
        n_clusters_per_class=2,
        random_state=42,
    )
    # sparsify towards a realistic CSR design matrix
    rng = np.random.RandomState(42)
    X[rng.rand(*X.shape) < 0.8] = 0.0
    return sparse.csr_matrix(X), y


def test_weighted_smote_large_csr_accounting():
    # the books balance on a CSR dataset with at least 20k input rows and at
    # most 40k output rows
    X, y = _make_sparse_data()
    assert X.shape[0] >= N_SAMPLES
    sample_weight = np.random.RandomState(0).rand(X.shape[0]) + 0.1

    result = SMOTE(random_state=42, k_neighbors=5).fit_resample_weighted(
        X, y, sample_weight
    )
    n_inputs = X.shape[0]
    n_outputs = result.X.shape[0]
    assert n_outputs <= MAX_OUTPUT_ROWS
    assert sparse.issparse(result.X) and result.X.format == "csr"

    result.check_conserved(sample_weight, rtol=RTOL, atol=ATOL)

    # kept rows are funded by themselves, synthetic rows by exactly two
    # endpoints of the same class
    y_in = np.asarray(y)
    y_out = np.asarray(result.y)
    input_index = result.contributions.input_index
    output_index = result.contributions.output_index
    np.testing.assert_array_equal(y_in[input_index], y_out[output_index])
    is_synthetic = output_index >= n_inputs
    np.testing.assert_array_equal(
        input_index[~is_synthetic], output_index[~is_synthetic]
    )
    n_funders = np.bincount(output_index[is_synthetic], minlength=n_outputs)[n_inputs:]
    np.testing.assert_array_equal(n_funders, 2)
    # positive weights everywhere: kept rows and non-degenerate synthetic
    # rows get a positive weight
    assert np.all(result.sample_weight[:n_inputs] > 0)
    assert np.all(result.sample_weight[n_inputs:] > 0)
    # the transfers stay sparse: far fewer records than n_inputs * n_outputs
    assert len(result.contributions) <= n_inputs + 2 * (n_outputs - n_inputs)


_BENCHMARK_SCRIPT = r"""
import resource
import sys

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification

mode = sys.argv[1]

X, y = make_classification(
    n_samples=20_000, n_features=30, n_informative=6, n_redundant=2,
    weights=[0.95, 0.05], flip_y=0.01, class_sep=1.5,
    n_clusters_per_class=2, random_state=42,
)
rng = np.random.RandomState(42)
X[rng.rand(*X.shape) < 0.8] = 0.0
X = sparse.csr_matrix(X)
sample_weight = np.random.RandomState(0).rand(X.shape[0]) + 0.1

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
if mode == "nn-only":
    # only the nearest-neighbour search shared by both variants
    smote._validate_estimator()
    target_class_indices = np.flatnonzero(y == 1)
    X_class = X[target_class_indices]
    smote.nn_k_.fit(X_class)
    smote.nn_k_.kneighbors(X_class, return_distance=False)
    n_out = -1
elif mode == "unweighted":
    X_res, y_res = smote.fit_resample(X, y)
    n_out = X_res.shape[0]
else:
    result = smote.fit_resample_weighted(X, y, sample_weight)
    result.check_conserved(sample_weight)
    n_out = result.X.shape[0]

peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"PEAK_KIB={peak_kib} N_OUT={n_out}")
"""


def _run_benchmark(mode):
    output = subprocess.run(
        [sys.executable, "-c", _BENCHMARK_SCRIPT, mode],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    metrics = dict(token.split("=") for token in output.split())
    return int(metrics["PEAK_KIB"]), int(metrics["N_OUT"])


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="ru_maxrss units differ on Windows"
)
def test_weighted_accounting_memory_overhead():
    pytest.importorskip("resource")
    peak_nn_only, _ = _run_benchmark("nn-only")
    peak_unweighted, n_out = _run_benchmark("unweighted")
    peak_weighted, n_out_weighted = _run_benchmark("weighted")

    assert n_out == n_out_weighted <= MAX_OUTPUT_ROWS
    # the neighbour search is common to both variants; the extra peak memory
    # of the weight accounting must stay within the budget
    overhead_mib = (peak_weighted - peak_unweighted) / 1024.0
    assert overhead_mib <= MEMORY_BUDGET_MIB, (
        f"weight accounting used {overhead_mib:.1f} MiB extra peak memory "
        f"(budget: {MEMORY_BUDGET_MIB} MiB; neighbour-search-only peak: "
        f"{peak_nn_only / 1024:.1f} MiB, unweighted peak: "
        f"{peak_unweighted / 1024:.1f} MiB, weighted peak: "
        f"{peak_weighted / 1024:.1f} MiB)"
    )
