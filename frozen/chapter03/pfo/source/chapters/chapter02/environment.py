"""Chapter-2 environment entry, bound to the fixed-period-total contract."""


def build_environment(domain, radio_groups=100, *, condition_file=None):
    from allocation23.environments import build_environment as shared_environment
    return shared_environment(domain, 2, radio_groups, condition_file=condition_file)


def make_baseline_environment(domain, **kwargs):
    if 'chapter' in kwargs:
        raise ValueError('The chapter-2 environment does not accept a chapter override')
    from allocation23.baselines.adapter import BaselineEnv
    return BaselineEnv(domain, 2, **kwargs)
