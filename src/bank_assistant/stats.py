"""Small statistics helpers used by every evaluation.

Eval sets here are small (106 BIRD questions, 682 loans), so a single number like "47%" hides
a lot of noise. Every reported metric gets a 95% bootstrap confidence interval, and
comparisons between two systems use a PAIRED bootstrap on the same questions, which is far
more sensitive than comparing two separate intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    value: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.value:.2f} [{self.low:.2f}, {self.high:.2f}]"

    def as_dict(self) -> dict:
        return {"value": self.value, "low": self.low, "high": self.high, "n": self.n}


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Estimate:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return Estimate(float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)), len(v))


@dataclass
class Comparison:
    diff: float
    low: float
    high: float
    p_value: float   # two-sided, share of bootstrap diffs on the other side of zero, doubled
    n: int

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
        return f"{self.diff:+.2f} [{self.low:+.2f}, {self.high:+.2f}], p={self.p_value:.3f}"


def paired_bootstrap(a, b, n_boot: int = 5000, seed: int = 0) -> Comparison:
    """Difference mean(b) - mean(a) on the SAME items (e.g. per-question correct/incorrect)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired comparison needs the same items in the same order"
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return Comparison(float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)),
                      float(min(p, 1.0)), len(d))


def bootstrap_metric(y, p, metric, n_boot: int = 2000, seed: int = 0) -> Estimate:
    """CI for a metric of predictions (e.g. ROC AUC), resampling the evaluated items."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():  # need both classes for AUC
            vals.append(metric(y[idx], p[idx]))
    return Estimate(float(metric(y, p)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), len(y))


def paired_bootstrap_metric(y, p_a, p_b, metric, n_boot: int = 2000, seed: int = 0) -> Comparison:
    """metric(b) - metric(a) on the same items, e.g. does model B have a higher AUC than model A?"""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            diffs.append(metric(y[idx], p_b[idx]) - metric(y[idx], p_a[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return Comparison(float(metric(y, p_b) - metric(y, p_a)), float(np.quantile(diffs, 0.025)),
                      float(np.quantile(diffs, 0.975)), float(min(p, 1.0)), len(y))
