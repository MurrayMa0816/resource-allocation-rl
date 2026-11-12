"""Absolute native-update eligibility indexed by real physical transitions.

Vector collection changes when a learner can execute a group of due updates,
but must not create seven extra updates at an eight-lane warmup boundary. The
shared worker may wait for replay readiness, then execute this clock's debt.
This pure helper neither samples replay nor performs an optimizer update.
"""
from numbers import Integral


def _integer(value,name,minimum):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def physical_update_target(train_steps,learning_starts,batch_size,updates_per_step):
    """Return total eligible updates, including the first warmup-boundary one.

    Eligibility is based only on original training interactions, excluding
    development and C-MORL internal selection. Replay readiness is a separate
    execution condition. If an n-step buffer becomes ready late, earlier due
    updates remain debt rather than silently disappearing.
    """
    steps=_integer(train_steps,'train_steps',0)
    starts=_integer(learning_starts,'learning_starts',0)
    batch=_integer(batch_size,'batch_size',1)
    utd=_integer(updates_per_step,'updates_per_step',1)
    return max(0,steps-max(starts,batch)+1)*utd
