"""Declared action-coordinate calibration for the native baseline tasks.

This module uses public feasibility and elapsed time, never rewards or future
paths. The generic learners retain their raw bounded controls in replay/Q.
PASPO/RPO retain allocation coordinates on the actual physical polytope.
"""
from __future__ import annotations

import math
import numpy as np

from ..methods.constraints import bounded_proportional

REVISION = 'native-baseline-action-calibration-full-spend-v2'
GENERIC_METHODS = ('td7', 'simba_v2', 'dime')


def action_family_for(method, chapter):
    if method not in ('paspo', 'rpo', *GENERIC_METHODS):
        raise ValueError('Unknown baseline method')
    return 'total_space' if int(chapter) == 3 and method in GENERIC_METHODS else 'polytope'


def reference_amount(state, horizon, lower=0.):
    """Public neutral amount scale: remaining allowance / remaining rounds."""
    period = int(round(float(np.asarray(state['global'])[0]) * int(horizon)))
    left = max(1, int(horizon) - period)
    return min(float(state['budget']), max(float(lower), float(state['remaining']) / left))


def amount_from_control(control, upper, reference, lower=0.):
    """Endpoint-complete odds map on [lower, upper] at the public reference.

    For zero lower, q/U = sigmoid(logit(reference/U) + atanh(z)). This is a fixed coordinate
    scale, with no learned head beyond the existing actor's first output.
    Exact endpoint controls reach zero and the full feasible amount. When the
    neutral amount equals U, a one-sided linear branch keeps all [0,U] reachable.
    """
    z, upper, reference, lower = map(float, (control, upper, reference, lower))
    if not (-1. <= z <= 1.) or not 0. <= lower <= reference <= upper:
        raise ValueError('Invalid amount coordinate or public scale')
    if lower == upper or z == -1.:
        return lower
    if z == 1.:
        return upper
    if z == 0.:
        return reference
    if lower:
        # Shift the original endpoint-complete odds coordinate to the same
        # public viable interval used by every method. No learner is changed.
        return min(upper, lower + amount_from_control(z, upper-lower, reference-lower))
    if upper == 0. or z == -1.:
        return 0.
    if z == 1.:
        return upper
    if z == 0.:
        return reference
    if reference == upper:
        return upper * (1. + min(z, 0.))
    if reference == 0.:
        return upper * max(z, 0.)
    r = reference / upper
    log_odds = math.log(r) - math.log1p(-r) + math.atanh(z)
    fraction = (1. / (1. + math.exp(-log_odds)) if log_odds >= 0.
                else math.exp(log_odds) / (1. + math.exp(log_odds)))
    return min(upper, upper * fraction)


def control_from_amount(amount, upper, reference, lower=0.):
    """Inverse representative used to verify full native action permissions."""
    amount, upper, reference, lower = map(float, (amount, upper, reference, lower))
    if not 0. <= lower <= amount <= upper or not lower <= reference <= upper:
        raise ValueError('Invalid feasible amount')
    if lower:
        return control_from_amount(amount-lower, upper-lower, reference-lower)
    if upper == 0. or amount == 0.:
        return -1.
    if reference == upper:
        return amount / upper - 1.
    if reference == 0.:
        return amount / upper
    if amount == upper:
        return 1.
    r, f = reference / upper, amount / upper
    return math.tanh(math.log(f) - math.log1p(-f) - math.log(r) + math.log1p(-r))


def neutral_allocation(caps, total):
    """Exchangeable equal-weight feasible configuration, without task ranking."""
    return bounded_proportional(np.ones_like(caps, dtype=float), caps, float(total))


def neutral_polytope_samples(caps, reference_fraction, parameters):
    """Reward-blind prior: amount Beta, then exchangeable spatial weights.

    The first coordinate is unused resource. The Beta mode for executed amount
    is the neutral reference fraction; remaining allocation weights are Gamma(8)
    draws with capped proportional completion. These are explicitly NOT volume
    uniform samples. PASPO fits its usual conditional Beta MLE to these samples.
    """
    caps = np.asarray(caps, dtype=np.float64)
    r = float(reference_fraction)
    if caps.ndim != 1 or len(caps) < 2 or not 0. < r < 1.:
        raise ValueError('Positive interior neutral fraction required')
    count = parameters['debias_chains'] * parameters['debias_draws']
    rng = np.random.default_rng(parameters['debias_seed'])
    concentration = 96.
    fractions = rng.beta(1. + concentration * r,
                         1. + concentration * (1. - r), size=count)
    spatial_caps = caps[1:]
    rows = []
    for q in fractions:
        weights = rng.gamma(8., size=len(spatial_caps))
        allocation = bounded_proportional(weights, spatial_caps, min(float(q), float(spatial_caps.sum())))
        rows.append(np.r_[1. - float(allocation.sum()), allocation])
    return np.asarray(rows, dtype=np.float64)


def neutral_context(env):
    """Only fixed public dimensions/scales used by geometric prior offsets."""
    fields={x['name']:x for x in env.get_contract()['observation_schema']}
    return dict(horizon=int(env.horizon),initial_budget=float(env.spec.initial_budget),
                global_start=fields['global']['start'],global_stop=fields['global']['stop'],
                constraint_unit=1.)


def reference_fraction_tensor(obs,target,context):
    """Read the same public time and remaining-budget coordinates as the NN."""
    import torch
    target=target.reshape(-1,1).double()
    phase=obs[:,context['global_start']:context['global_start']+1].double()
    remaining=obs[:,context['global_stop']-1:context['global_stop']].double()*context['initial_budget']
    left=(context['horizon']-torch.round(phase*context['horizon'])).clamp_min(1.)
    wanted=remaining.clamp_min(0.)/left
    denominator=torch.where(target>0.,target,torch.ones_like(target))
    return torch.where(target>0.,wanted/denominator,0.).clamp(0.,1.)


def neutral_mean_tensor(obs,caps,target,context):
    """RPO's state-only initial mean on its unchanged physical basic box."""
    import torch
    caps,target=caps.double(),target.reshape(-1,1).double()
    amount=target*reference_fraction_tensor(obs,target*context['constraint_unit'],context)
    physical_caps=caps[:,1:]
    allocation=torch.zeros_like(physical_caps)
    active=physical_caps>0.
    remaining=amount
    # Exchangeable capped equal shares. In the current native tasks positive
    # caps are equal, so this normally completes in one vectorized iteration.
    for _ in range(physical_caps.shape[-1]+1):
        share=remaining/active.sum(-1,keepdim=True).clamp_min(1)
        hit=active & (physical_caps<=share)
        if not bool(hit.any()):
            allocation=allocation+active*share
            break
        allocation=torch.where(hit,physical_caps,allocation)
        active=active & ~hit
        remaining=(amount-allocation.sum(-1,keepdim=True)).clamp_min(0.)
    augmented=torch.cat(((target-amount).clamp_min(0.),allocation),-1)
    denominator=torch.where(caps[:,:-1]>0.,caps[:,:-1],torch.ones_like(caps[:,:-1]))
    ratio=torch.where(caps[:,:-1]>0.,augmented[:,:-1]/denominator,.5)
    return torch.atanh((2.*ratio-1.).clamp(-1.+1e-7,1.-1e-7)).float()
