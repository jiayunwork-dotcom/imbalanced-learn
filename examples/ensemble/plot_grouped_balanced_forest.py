"""
==========================================================
Group-aware sampling for unseen-device evaluation
==========================================================

A device usually leaves several rows in the data set.  When a forest trains on
some rows of a device, scoring the other rows of the same device as "out of
bag" overstates the generalisation to an unseen device.  Passing a ``groups``
identifier to :class:`~imblearn.ensemble.BalancedRandomForestClassifier` keeps
whole devices together: every tree trains on the rows of the selected devices
only and can only vote for devices it has never seen.

This example compares the historical row-wise sampling with the grouped
sampling on a synthetic data set where each device carries a drift that a tree
can memorise.

"""

# Authors: imbalanced-learn developers
# License: MIT

# %%
print(__doc__)

# %%
# Synthetic devices
# -----------------
#
# Every device has a small random offset added to both features, so rows of one
# device resemble each other.  The label depends on the features plus a
# per-device effect, which makes memorising a device profitable on its own
# rows.  The class ratio is imbalanced, and a few devices mix both labels.
import numpy as np

rng = np.random.RandomState(0)
n_devices = 60
rows_per_device = 8
n_samples = n_devices * rows_per_device

device = np.repeat(np.arange(n_devices), rows_per_device)
device_offset = rng.normal(scale=1.5, size=(n_devices, 2))[device]
X = rng.normal(size=(n_samples, 2)) + device_offset
logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(scale=0.5, size=n_samples)
y = (logits > 0.8).astype(int)

# A handful of devices mix the two labels.
mixed_devices = rng.choice(n_devices, size=8, replace=False)
for chosen in mixed_devices:
    rows = np.flatnonzero(device == chosen)
    y[rows[rng.permutation(rows.size) == 0]] = 1 - y[rows[0]]

print(f"class counts: {np.bincount(y)}")
print(f"devices: {n_devices}, rows per device: {rows_per_device}")

# %%
# Row-wise sampling (historical behaviour)
# ----------------------------------------
#
# ``RandomUnderSampler`` selects individual rows.  Several rows of the same
# device can end up in the training set while its remaining rows are scored,
# which is the leakage this example illustrates.
from collections import Counter

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)
X_rows, y_rows = rus.fit_resample(X, y)
print(f"rows kept by the row-wise sampler: {Counter(y_rows)}")
partially_kept = [
    d
    for d in np.unique(device)
    if 0 < np.isin(np.flatnonzero(device == d), rus.sample_indices_).sum()
    < np.count_nonzero(device == d)
]
print(f"devices with some rows kept and some rows dropped: {len(partially_kept)}")

# %%
# Group-aware sampling
# --------------------
#
# The grouped sampler keeps or drops each device as a whole.  Each class targets
# the size of the smallest class; here a few devices mix labels, so the balance
# is the best one reachable with whole devices rather than an exact match.
rus_grouped = RandomUnderSampler(random_state=0)
X_devices, y_devices = rus_grouped.fit_resample(X, y, groups=device)
print(f"rows kept by the grouped sampler: {Counter(y_devices)}")
print(f"devices kept: {len(rus_grouped.group_ids_)} of {rus_grouped.n_groups_}")
kept_rows = np.isin(device, rus_grouped.group_ids_)
same_rows = np.array_equal(
    kept_rows, np.isin(np.arange(n_samples), rus_grouped.sample_indices_)
)
print(f"all rows of the selected devices are kept and the others dropped: {same_rows}")

# %%
# Out-of-bag estimates of both forests
# ------------------------------------
#
# The grouped forest can compute out-of-bag scores: a tree never votes for a
# device it trained on.  ``oob_counts_`` tells how many trees could vote for
# each row, and rows without any vote are skipped by ``oob_score_``.
from imblearn.ensemble import BalancedRandomForestClassifier

params = dict(
    n_estimators=100,
    random_state=0,
    sampling_strategy="auto",
    replacement=False,
    bootstrap=False,
    oob_score=True,
)

grouped_forest = BalancedRandomForestClassifier(**params)
grouped_forest.fit(X, y, groups=device)
votes = grouped_forest.oob_counts_
print(f"rows with at least one eligible tree: {(votes > 0).sum()} of {n_samples}")
print(f"grouped out-of-bag accuracy: {grouped_forest.oob_score_:.3f}")

row_params = dict(params, bootstrap=True)
row_forest = BalancedRandomForestClassifier(**row_params)
row_forest.fit(X, y)
print(f"row-wise out-of-bag accuracy: {row_forest.oob_score_:.3f}")

# %%
# The row-wise score is computed on rows whose device was seen during training
# by most trees, whereas the grouped score only uses devices that were left out
# of the tree.  The number of rows with a vote is smaller in grouped mode: a
# tree only leaves out the devices that the balanced target does not require.
# To estimate the accuracy on a genuinely unseen device, hold out whole devices
# and compare the two modes on that split.
held_out = np.isin(device, np.arange(12))
X_train, y_train = X[~held_out], y[~held_out]
devices_train = device[~held_out]
X_test, y_test = X[held_out], y[held_out]

grouped_forest = BalancedRandomForestClassifier(**params)
grouped_forest.fit(X_train, y_train, groups=devices_train)

row_forest = BalancedRandomForestClassifier(**row_params)
row_forest.fit(X_train, y_train)

from sklearn.metrics import balanced_accuracy_score

print(
    "balanced accuracy on unseen devices -- grouped fit: "
    f"{balanced_accuracy_score(y_test, grouped_forest.predict(X_test)):.3f}"
)
print(
    "balanced accuracy on unseen devices -- row-wise fit: "
    f"{balanced_accuracy_score(y_test, row_forest.predict(X_test)):.3f}"
)

# %%
# Inspecting the per-tree choices
# -------------------------------
#
# ``samplers_`` exposes, for each tree, the selected devices and the exact rows
# they kept.  Different trees choose different devices while reaching the same
# class counts, which is what gives the out-of-bag estimate its coverage.
selections = {tuple(sorted(sampler.group_ids_)) for sampler in grouped_forest.samplers_}
print(f"distinct device selections across the trees: {len(selections)}")
for index in range(3):
    sampler = grouped_forest.samplers_[index]
    print(
        f"tree {index}: {len(sampler.group_ids_)} devices, "
        f"{len(sampler.sample_indices_)} rows, "
        f"class counts {np.bincount(y_train[sampler.sample_indices_], minlength=2)}"
    )

# %%
# Every selected row belongs to a selected device and no row is duplicated.
sampler = grouped_forest.samplers_[0]
print(
    "sampler rows are the rows of its devices:",
    np.array_equal(
        sampler.sample_indices_,
        np.flatnonzero(np.isin(devices_train, sampler.group_ids_)),
    ),
)
print(
    "no duplicated row:",
    len(sampler.sample_indices_) == len(np.unique(sampler.sample_indices_)),
)
