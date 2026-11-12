"""Fixed feasibility maps and complete latent Gaussian log likelihood.

These functions contain no trainable parameters or task-specific ranking.
"""
import math
import numpy as np

LOG2PI = math.log(2. * math.pi)


def _strict_upper(action, upper):
    """Correct only floating-point positive excess, keeping all coordinates legal."""
    a = np.asarray(action, dtype=np.float64).copy()
    u, total = float(upper), float(a.sum())
    if total > u:
        tolerance = 512*np.finfo(float).eps*max(total, u, np.finfo(float).tiny)
        if total-u > tolerance:
            raise ArithmeticError('Non-roundoff total excess')
        if u == 0:
            return np.zeros_like(a)
        a *= np.nextafter(u/total, 0.)
        for _ in range(4):
            excess = float(a.sum())-u
            if excess <= 0:
                break
            j = int(a.argmax())
            a[j] = max(0., np.nextafter(a[j]-excess, 0.))
        if float(a.sum()) > u:
            raise ArithmeticError('Strict total correction failed')
    return a

def bounded_proportional(weights, caps, total):
    """Generic continuous bounded proportional allocation; no business ranking."""
    w, c = np.asarray(weights, float).copy(), np.asarray(caps, float)
    q = float(total)
    if (w.ndim != 1 or c.shape != w.shape or not np.isfinite(w).all()
            or not np.isfinite(c).all() or np.any(w < 0) or np.any(c < 0)
            or not math.isfinite(q) or q < 0 or q > float(c.sum())):
        raise ValueError('Nonnegative finite feasible allocation inputs required')
    if q == 0:
        return np.zeros_like(c)
    if q == float(c.sum()):
        return _strict_upper(c, q)
    active = c > 0
    if not np.any(w[active] > 0):
        raise ValueError('Positive active allocation weight required')
    w = np.maximum(w, np.finfo(float).tiny)
    w /= w.max()
    a = np.zeros_like(c)
    remaining = q
    for _ in range(len(c)+1):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        p = remaining*w[ids]/w[ids].sum()
        hit = p >= c[ids]
        if not np.any(hit):
            a[ids] = p
            break
        used = ids[hit]
        a[used] = c[used]
        active[used] = False
        remaining = q-float(a.sum())
    tolerance = 512*np.finfo(float).eps*max(q, np.finfo(float).tiny)
    delta = q-float(a.sum())
    if abs(delta) > tolerance:
        raise ArithmeticError('Proportional map did not conserve requested total')
    if delta > 0:
        j = int(np.argmax(c-a))
        a[j] += min(delta, c[j]-a[j])
    return _strict_upper(np.minimum(a, c), q)

def capped_simplex(proposal, caps, total):
    """Euclidean bounded-simplex projection using piecewise-linear breakpoints.

    Incremental free-coordinate updates avoid subtracting a tiny quota from a
    large Lagrange multiplier. A separated top group handles sub-ULP quotas.
    """
    p, c, q = np.asarray(proposal, float), np.asarray(caps, float), float(total)
    if q <= 0:
        return np.zeros_like(c)
    if q >= float(c.sum()):
        return _strict_upper(c, q)
    eligible = c > 0
    top = float(p[eligible].max())
    best = eligible & (p == top)
    others = eligible & ~best
    if float(c[best].sum()) >= q and (not np.any(others) or top-float(p[others].max()) >= q):
        result = np.zeros_like(c)
        result[best] = bounded_proportional(np.ones(best.sum()), c[best], q)
        return result
    indices = np.flatnonzero(eligible)
    points = np.concatenate((p[indices], p[indices]-c[indices]))
    event_ids = np.concatenate((indices, indices))
    starts = np.concatenate((np.ones(len(indices), bool), np.zeros(len(indices), bool)))
    order = np.argsort(-points, kind='stable')
    a = np.zeros_like(c)
    free = np.zeros(len(c), bool)
    current = float(points[order[0]])
    j = 0
    while j < len(order):
        point = float(points[order[j]])
        count = int(free.sum())
        used = float(a.sum())
        if count:
            delta = (q-used)/count
            if delta <= current-point:
                a[free] += max(0., delta)
                return _strict_upper(np.minimum(a, c), q)
            a[free] += current-point
        current = point
        while j < len(order) and float(points[order[j]]) == point:
            k = order[j]
            i = event_ids[k]
            if starts[k]:
                free[i] = True
            else:
                a[i] = c[i]
                free[i] = False
            j += 1
    if abs(float(a.sum())-q) > 512*np.finfo(float).eps*max(q, np.finfo(float).tiny):
        raise ArithmeticError('Bounded-simplex projection did not reach its total')
    return _strict_upper(a, q)

def gaussian_logprob(raw, mean, sigma):
    return (-.5*((raw-mean)/sigma).square()-sigma.log()-.5*LOG2PI).sum(-1, keepdim=True)
