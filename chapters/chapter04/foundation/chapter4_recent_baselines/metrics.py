"""Explicit sample-efficiency metrics with failed attainment kept censored.

All x coordinates count newly consumed real environment steps, including
warmup; replay samples and optimizer updates never increment this coordinate.
Evaluation is a separate, fixed-path stream. Thresholds must be preregistered
from a common reference/protocol, not chosen per method from held-out scores.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def _curve(steps, scores):
    x, y = np.asarray(steps, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if (x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(x < 0) or np.any(np.diff(x) <= 0)):
        raise ValueError('Finite equal-length curves on strictly increasing real-step coordinates required')
    return x, y


def sustained_attainment(steps, scores, target: float, *, consecutive: int = 2,
                         budget: float | None = None):
    """N_q is the real-step cost when sustained attainment is confirmed.

    For two consecutive passing evaluations, N_q is the SECOND evaluation's
    real-step coordinate. A failed run has N_q=None and right_censored=True;
    its last step is a censoring time, never a fabricated attainment time.
    """
    x, y = _curve(steps, scores)
    if not np.isfinite(target):
        raise ValueError('target must be finite')
    if isinstance(consecutive, bool) or int(consecutive) != consecutive or consecutive < 1:
        raise ValueError('consecutive must be a positive integer')
    if budget is None:
        budget = float(x[-1])
    if not np.isfinite(budget) or budget < 0:
        raise ValueError('budget must be finite and nonnegative')
    eligible = x <= budget
    x, y = x[eligible], y[eligible]
    if not len(x):
        raise ValueError('No evaluations at or before the declared budget')
    streak = 0
    for i, score in enumerate(y):
        streak = streak + 1 if score >= target else 0
        if streak >= consecutive:
            return dict(N_q=float(x[i]), first_passing_step=float(x[i-consecutive+1]),
                        right_censored=False, attained=True, observation_end=float(x[-1]),
                        declared_budget=float(budget), target=float(target), consecutive=int(consecutive))
    return dict(N_q=None, first_passing_step=None, right_censored=True, attained=False,
                observation_end=float(x[-1]), declared_budget=float(budget),
                target=float(target), consecutive=int(consecutive))


# Convenient unambiguous alias used by experiment report scripts.
first_attainment = sustained_attainment


def attainment_curve(results: Iterable[dict], step_grid):
    """Fraction of independent seeds with confirmed N_q by each budget.

    A seed whose observation ended before a requested budget is not assumed to
    have failed indefinitely: report the known lower/upper bounds separately.
    """
    rows = list(results)
    grid = np.asarray(step_grid, dtype=np.float64)
    if not rows or grid.ndim != 1 or not np.isfinite(grid).all() or np.any(np.diff(grid) < 0):
        raise ValueError('Nonempty results and a finite ordered step grid required')
    lower, upper, known = [], [], []
    for cost in grid:
        reached = sum(r['N_q'] is not None and r['N_q'] <= cost for r in rows)
        unknown = sum(r['right_censored'] and r['observation_end'] < cost for r in rows)
        lower.append(reached/len(rows))
        upper.append((reached+unknown)/len(rows))
        known.append(len(rows)-unknown)
    return dict(real_steps=grid.tolist(), attained_fraction=lower,
                possible_attained_fraction=upper, seeds_observed_or_attained=known, seeds=len(rows))


def normalized_auc(steps, scores, *, start: float | None = None, end: float | None = None):
    """Trapezoidal learning-curve area divided by its observed real-step span.

    The result retains score units. Only interpolation inside actually observed
    endpoints is allowed. No artificial score at step zero and no best-so-far
    curve are inserted. Use the same start/end window across all methods.
    """
    x, y = _curve(steps, scores)
    if len(x) < 2:
        raise ValueError('AUC requires at least two evaluations')
    start = float(x[0] if start is None else start)
    end = float(x[-1] if end is None else end)
    if not np.isfinite([start, end]).all() or start < x[0] or end > x[-1] or end <= start:
        raise ValueError('AUC window must lie within the observed curve endpoints')
    inside = (x > start) & (x < end)
    xx = np.concatenate(([start], x[inside], [end]))
    yy = np.interp(xx, x, y)
    area = float(np.sum(.5 * (yy[:-1] + yy[1:]) * np.diff(xx)))
    return dict(normalized_auc=area/(end-start), area=area, start=start, end=end,
                score_units='same as supplied scores', interpolation='piecewise_linear_observed_only')


def summarize_attainment(results: Iterable[dict], *, budget: float):
    """Summarize a fixed, fully observed horizon without hiding failed seeds."""
    rows = list(results)
    if not rows or not np.isfinite(budget) or budget <= 0:
        raise ValueError('Nonempty results and a positive budget required')
    if any(r['observation_end'] < budget and (r['N_q'] is None or r['N_q'] > budget) for r in rows):
        raise ValueError('Every seed must be observed to the summary budget or have attained earlier')
    times = [float(r['N_q']) for r in rows if r['N_q'] is not None and r['N_q'] <= budget]
    restricted = [min(float(r['N_q']), budget) if r['N_q'] is not None else float(budget) for r in rows]
    return dict(seeds=len(rows), attained_seeds=len(times), censored_seeds=len(rows)-len(times),
                attained_fraction=len(times)/len(rows), budget=float(budget),
                restricted_mean_steps=float(np.mean(restricted)),
                median_steps_among_attained=float(np.median(times)) if times else None,
                note='Restricted mean caps unresolved seeds at budget; it is not an uncensored mean N_q.')
