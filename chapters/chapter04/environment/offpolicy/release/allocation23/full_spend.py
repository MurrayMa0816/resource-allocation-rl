"""Public, robust feasibility of allocating the entire Chapter 3 budget.

No future channel, arrivals, conversion rates or success draws are read here.
Radio can confirm at most one packet per active receiver and period. Assuming
every possible current/future confirmation succeeds gives the smallest future
physical spending capacity. Keeping the remaining budget below that capacity
guarantees a feasible all-spent continuation without funding completed users.
"""
from __future__ import annotations
import numpy as np

REVISION = 'chapter3-full-budget-allocation-public-viability-v1'


def budget_roundoff(initial_budget):
    """Fixed floating-point tolerance, not a positive unallocated allowance."""
    return 512*np.finfo(np.float64).eps*max(abs(float(initial_budget)),np.finfo(float).tiny)


def terminal_budget_tolerance(spec):
    return min(float(spec.action_tolerance),budget_roundoff(spec.initial_budget))


def radio_future_capacity(confirmed_packets,source_packets,rounds_left,per_terminal_capacity):
    """Guaranteed capacity AFTER this action under fastest legal completion."""
    counts=np.asarray(confirmed_packets)
    if (counts.ndim!=1 or not np.isfinite(counts).all() or
            np.any(counts!=np.rint(counts)) or np.any(counts<0) or
            np.any(counts>source_packets) or rounds_left<1 or
            int(rounds_left)!=rounds_left or per_terminal_capacity<=0):
        raise ValueError('Valid public packet counts, horizon and physical capacity required')
    left=int(source_packets)-counts.astype(np.int64)
    periods=np.minimum(np.maximum(left-1,0),int(rounds_left)-1)
    return float(periods.sum(dtype=np.int64))*float(per_terminal_capacity)


def radio_minimum_input(remaining,confirmed_packets,source_packets,rounds_left,
                        per_terminal_capacity,*,tolerance=0.):
    """q >= R-F preserves a feasible fully allocated continuation for all draws."""
    remaining=float(remaining)
    if not np.isfinite(remaining) or remaining<0:
        raise ValueError('Nonnegative finite remaining budget required')
    counts=np.asarray(confirmed_packets)
    future=radio_future_capacity(counts,source_packets,rounds_left,per_terminal_capacity)
    current=float(np.count_nonzero(counts<int(source_packets)))*float(per_terminal_capacity)
    if remaining>current+future+float(tolerance):
        raise ArithmeticError('Remaining radio budget cannot fit any guaranteed full-allocation continuation')
    lower=max(0.,remaining-future)
    upper=min(remaining,current)
    if lower>upper+float(tolerance):
        raise ArithmeticError('Infeasible radio full-allocation interval')
    return min(lower,upper)


def full_spend_bounds(state,spec):
    """Return the common current total interval using existing public state."""
    upper=float(np.asarray(state['budget']))
    if spec.chapter!=3:return 0.,upper
    remaining=float(np.asarray(state['remaining']))
    phase=float(np.asarray(state['global'])[0])
    period=int(round(phase*spec.horizon));left=spec.horizon-period
    tolerance=terminal_budget_tolerance(spec)
    if period<0 or period>spec.horizon or remaining<0 or upper<0:
        raise ValueError('Invalid public phase/resource state')
    if left==0:
        if remaining>tolerance:raise ArithmeticError('Full-allocation task ended with unspent resource')
        return 0.,0.
    if spec.study_domain=='v5':
        lower=remaining if left==1 else 0.
    else:
        local=np.asarray(state['local'])
        if local.shape!=(spec.groups,spec.local_dim):raise ValueError('Public local state shape mismatch')
        confirmed=np.rint(local[:,2]*spec.source_packets).astype(np.int64)
        lower=radio_minimum_input(remaining,confirmed,spec.source_packets,left,
            spec.per_object_cap,tolerance=tolerance)
    if lower>upper+tolerance:raise ArithmeticError('Public current full-allocation interval is empty')
    return min(lower,upper),upper


def full_spend_lower(state,spec):
    return full_spend_bounds(state,spec)[0]


def assert_terminal_full_spend(remaining,spec):
    if spec.chapter==3 and abs(float(remaining))>terminal_budget_tolerance(spec):
        raise ArithmeticError('Chapter 3 must allocate the full initial resource within roundoff')
