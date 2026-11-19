"""Complete-episode dual terminal labels and frozen finite-horizon traces."""
from __future__ import annotations

import numpy as np
import torch


def terminal_labels(batches, groups, tau, device='cpu'):
    labels = []
    for batch in batches:
        quality = np.asarray(batch['terminal_quality'], dtype=np.float64)
        if quality.shape != (groups,) or not np.isfinite(quality).all():
            raise ValueError('One finite real terminal quality per object required')
        k = int((quality >= tau).sum())
        if (np.asarray(batch['K']).size != 1 or float(batch['K']) != k
                or not np.isclose(float(batch['C']), k / groups, rtol=0, atol=1e-12)):
            raise ValueError('Coverage must count actual terminal qualities at fixed tau')
        native = float(quality.mean())
        rewards, returns = np.asarray(batch['rewards']), np.asarray(batch['returns'])
        if (not np.isclose(native, rewards[-1], rtol=0, atol=1e-10)
                or not np.allclose(returns, native, rtol=0, atol=1e-10)):
            raise ValueError('Original native quality reward and MC returns must remain unchanged')
        labels.append((k / groups, native))
    return torch.as_tensor(labels, dtype=torch.float64, device=device)


@torch.no_grad()
def dual_terminal_trace(values, labels, log_rhos, mask, lengths):
    """Gamma=lambda=rho_cap=c_cap=1, independently for C and J.

    The local importance multiplier is supplied once in the policy surrogate.
    An inactive action has rho=1; it never cuts the physical episode.
    No current quality or intermediate reward is constructed.
    """
    if values.shape != (sum(lengths), 2) or labels.shape != (len(lengths), 2):
        raise ValueError('Two values per state and two labels per complete path required')
    if any(int(h) != h or h <= 0 for h in lengths):
        raise ValueError('Complete positive trajectory lengths required')
    logs, masks = log_rhos.reshape(-1), mask.reshape(-1)
    if logs.shape != (len(values),) or masks.shape != logs.shape:
        raise ValueError('One complete-event likelihood ratio/mask per decision required')
    if not all(torch.isfinite(x).all() for x in (values, labels, logs, masks)):
        raise FloatingPointError('Finite dual trace inputs required')
    rho = torch.where(masks > 0, logs.clamp_max(0).exp(), torch.ones_like(logs))
    targets, advantages = torch.empty_like(values), torch.empty_like(values)
    offset = 0
    for label, length in zip(labels, lengths):
        following = label
        for t in range(offset + int(length) - 1, offset - 1, -1):
            advantages[t] = following - values[t]
            targets[t] = (1 - rho[t]) * values[t] + rho[t] * following
            following = targets[t]
        offset += int(length)
    return targets.detach(), advantages.detach(), rho.detach()


def scalarize_advantages(advantages, weights, mask):
    """Mix raw-unit advantages first, then one common scalar normalization."""
    if advantages.ndim != 2 or advantages.shape[1] != 2:
        raise ValueError('Dual advantage columns (C,J) required')
    if weights.shape != (len(advantages), 1) or mask.shape != weights.shape:
        raise ValueError('Step-aligned scalar preferences/masks required')
    if not bool(((weights >= 0) & (weights <= 1)).all()):
        raise ValueError('Preferences in [0,1] required')
    raw = weights * advantages[:, :1] + (1 - weights) * advantages[:, 1:]
    active = mask > 0
    scale = raw[active].std(unbiased=False) if bool(active.any()) else raw.new_tensor(0.)
    mean = raw[active].mean() if bool(active.any()) else raw.new_tensor(0.)
    normalized = ((raw - mean) / (scale + 1e-8)).detach() if bool(active.any()) else raw.detach()
    return normalized, raw.detach(), float(scale), float(mean)
