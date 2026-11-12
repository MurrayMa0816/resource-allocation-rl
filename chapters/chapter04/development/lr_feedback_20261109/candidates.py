"""Development-only LR variants around the unchanged complete-trajectory F1.

The parent update commits every native SGD step before proposing the next LR.
This wrapper changes that next proposal only.  The low fixed-rate arm also
starts its warmup at 1e-4; the three other arms start at the native 3e-4.
No parent source, clip width, sampling rule or mathematical target is changed.
"""
from __future__ import annotations

import math
from allocation_efficiency.frequent import FrequentPPO, EPSILON, ACTOR_LR_BOUNDS

METHODS = ('original_feedback', 'fixed_3e4', 'fixed_1e4', 'relaxed_feedback')
SETTINGS = {
    'original_feedback': dict(initial_lr=3e-4, lower=EPSILON/4,
                              upper=EPSILON/2, factor=1.03, fixed=False),
    'fixed_3e4': dict(initial_lr=3e-4, lower=None, upper=None,
                      factor=1., fixed=True),
    'fixed_1e4': dict(initial_lr=1e-4, lower=None, upper=None,
                      factor=1., fixed=True),
    'relaxed_feedback': dict(initial_lr=3e-4, lower=.05, upper=.10,
                             factor=1.03, fixed=False),
}


def next_rate(method, lr, change):
    """Strict inequalities match the original rule; equality holds the LR."""
    if method not in SETTINGS:
        raise ValueError('Unknown development LR arm')
    if not math.isfinite(lr) or not ACTOR_LR_BOUNDS[0] <= lr <= ACTOR_LR_BOUNDS[1]:
        raise ValueError('Learning rate outside native bounds')
    if not math.isfinite(change) or change < 0:
        raise ValueError('Finite nonnegative empirical change required')
    c = SETTINGS[method]
    if c['fixed']:
        return c['initial_lr'], 'fixed_no_feedback'
    if change > c['upper']:
        value, action = lr/c['factor'], 'decrease'
    elif change < c['lower']:
        value, action = lr*c['factor'], 'increase'
    else:
        value, action = lr, 'hold'
    return min(max(value, ACTOR_LR_BOUNDS[0]), ACTOR_LR_BOUNDS[1]), action


class DevelopmentLR(FrequentPPO):
    def __init__(self, spec, event_dim, seed, device, *, method):
        if method not in METHODS:
            raise ValueError('Unknown development arm')
        self.development_method = method
        super().__init__(spec, event_dim, seed, device, replay=True)
        self.controller['actor_lr'] = SETTINGS[method]['initial_lr']

    def metadata(self):
        result = super().metadata()
        c = SETTINGS[self.development_method]
        result.update(development_only=True,
                      development_version='lr-feedback-development-20261109-v1',
                      development_method=self.development_method,
                      actor_initial_lr=c['initial_lr'],
                      tv_lower_target=c['lower'], tv_target=c['upper'],
                      tv_alpha=c['factor']-1 if not c['fixed'] else None,
                      tv_beta=c['lower']/c['upper'] if not c['fixed'] else None,
                      tv_controller='disabled; change remains diagnostic' if c['fixed'] else 'native next-batch proposal with registered independent band',
                      generalized_clip=EPSILON,
                      clip_independent_of_controller_band=True,
                      scientific_role='fresh DEV only; no TEST or formal replacement')
        return result

    def update(self, actor, critic, ao, vo, batches, rng):
        c = SETTINGS[self.development_method]
        entered_lr = float(self.controller['actor_lr'])
        if c['fixed'] and entered_lr != c['initial_lr']:
            raise ValueError('Fixed-rate arm entered update at wrong rate')
        stats = super().update(actor, critic, ao, vo, batches, rng)
        # Warmup deliberately retains its entered LR in all four arms.
        if not stats['warmup'] and self.development_method != 'original_feedback':
            proposal = stats['actor_lr_after']
            rate, action = next_rate(self.development_method,
                                     stats['actor_lr_before'], stats['tv'])
            self.controller.update(actor_lr=rate, last_action=action)
            if c['fixed']:
                self.controller['calls'] = 0
            for group in ao.param_groups:
                group['lr'] = rate
            stats.update(actor_lr_after=rate,
                         tv_lr_factor=rate/stats['actor_lr_before'],
                         tv_controller_action=action,
                         tv_controller_ran=not c['fixed'],
                         tv_controller_calls=self.controller['calls'],
                         diagnostic_original_next_lr_proposal=proposal)
        stats.update(development_method=self.development_method,
                     tv_target=c['upper'], tv_lower_target=c['lower'],
                     feedback_band_independent_of_clip=True,
                     generalized_clip=EPSILON,
                     actor_initial_lr=c['initial_lr'],
                     tv_used_for_lr=not c['fixed'] and not stats['warmup'])
        return stats

    def load_state_dict(self, state):
        raise ValueError('Development runs start from zero and save inference weights only; optimizer/replay resume is disabled')
