"""Direct total-coordinate control; immutable native spatial policy is inherited.

Only the total decoder is changed: q = L + (U-L)*sigmoid(v). A fixed,
state-independent output bias matches the DEFAULT initial pacing fraction.
There is no per-state pacing/logit offset in this decoder.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
from allocation23.methods.policy import Actor

SCHEMA = 'chapter3-A-direct-native-supplement-v1'
ARM = 'A_DIRECT_TOTAL'


def initial_coordinate(spec):
    """Use static training geometry only, without sampling a task or test path."""
    if spec.chapter != 3:
        raise ValueError('Direct total control is a Chapter 3 arm')
    remaining = float(spec.initial_budget)
    if spec.study_domain == 'v5':
        upper = remaining if spec.peak_total is None else min(remaining, float(spec.peak_total))
        lower = remaining if spec.horizon == 1 else 0.
    else:
        capacity = int(spec.groups)*float(spec.per_object_cap)
        upper = min(remaining, capacity)
        future = min(int(spec.source_packets)-1, int(spec.horizon)-1)*capacity
        lower = max(0., remaining-future)
    reference = min(upper, max(lower, remaining/spec.horizon))
    fraction = (reference-lower)/(upper-lower) if upper > lower else .5
    if not 0. < fraction < 1.:
        raise ValueError('Registered default training initial geometry needs an interior reference')
    bias = math.log(fraction)-math.log1p(-fraction)
    return dict(initial_bias=bias, initial_fraction=fraction, initial_lower=lower,
                initial_upper=upper, initial_reference=reference,
                rule='one static scalar bias from training-spec initial public feasibility; not recalibrated by state, path or condition',
                parameterization='q=L+(U-L)*sigmoid(v); v=quota_head(state)',
                state_dependent_pacing_offset=False,
                initial_match_scope='deterministic total at default t=0 only; no claim of matched trajectory after the first action')


class DirectTotalActor(Actor):
    def __init__(self, spec, seed, *, initial_bias=None):
        super().__init__(spec, 'A_FULL', 'adaptive', seed, quota_reference='pace')
        self.direct_initialization = initial_coordinate(spec) if initial_bias is None else dict(
            initial_bias=float(initial_bias), rule='fixed scalar restored from training identity; never recalibrated during evaluation')
        bias = self.direct_initialization['initial_bias']
        if not math.isfinite(bias):
            raise ValueError('Finite direct coordinate initialization required')
        with torch.no_grad():
            self.quota_head[-1].bias.fill_(bias)

    def _full_spend_quota(self, state, eta=None):
        if eta is None:
            raise ValueError('Recorded direct total Gaussian event required')
        data = self._prepare(state)
        period = self._round(data)
        lower, upper = self._full_spend_interval(data)
        coordinate = torch.as_tensor(eta, dtype=upper.dtype, device=upper.device).reshape(-1)
        if coordinate.shape != upper.shape or not torch.isfinite(coordinate).all():
            raise ValueError('One finite direct total coordinate per state required')
        requested = lower+(upper-lower)*torch.sigmoid(coordinate)
        requested = torch.maximum(lower, torch.minimum(requested, upper))
        return torch.where(period < self.spec.horizon, requested, torch.zeros_like(upper))

    def metadata(self):
        result = super().metadata()
        result.update(method_version=SCHEMA, method_arm=ARM,
            quota_transform='L+(U-L)*sigmoid(v)',
            quota_reference='none; inherited compatibility attribute is unused by direct decoder',
            direct_initialization=self.direct_initialization,
            temporal_information='Same complete current-task context, time and budget inputs; direct learned feasible fraction without a state-dependent reference offset')
        return result


def _model_file(training_run):
    path = Path(training_run).expanduser().resolve()
    return path if path.is_file() else path/'model'/'state.pt'


def load_policy(training_run, target_task, device='cpu'):
    """Return a native actor with .sample(state,deterministic=True) and .act(state).

    Callers create the target Task (including any explicitly registered physical
    condition). The fixed training bias is restored unchanged. Legacy A models
    are rejected here and continue to use their unmodified legacy loader.
    """
    payload = torch.load(_model_file(training_run), map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or not payload.get('evaluation_only'):
        raise ValueError('This loader requires an explicit Direct evaluation payload')
    identity = payload['identity']
    if identity['method'] != ARM or not identity['direct_mapping'].get('state_dependent_pacing_offset') is False:
        raise ValueError('Direct identity missing or incompatible')
    source_spec, spec = identity['spec'], target_task.spec
    for key in ('domain', 'chapter', 'groups', 'local_dim', 'global_dim', 'source_packets', 'feature_version'):
        if source_spec[key] != getattr(spec, key):
            raise ValueError('Incompatible direct policy task structure: '+key)
    actor = DirectTotalActor(spec, identity['arguments']['seed'],
        initial_bias=identity['direct_mapping']['initial_bias'])
    actor.load_state_dict(payload['actor'], strict=True)
    actor.to(device=device, dtype=torch.float64)
    actor.eval()
    return actor
