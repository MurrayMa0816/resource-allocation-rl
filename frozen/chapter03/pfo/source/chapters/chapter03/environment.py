"""Chapter 3 environment entry, with the shared physics and public state."""


def make(domain, **kwargs):
    from allocation23.chapter3_baselines.environment import PublicBudgetEnv
    return PublicBudgetEnv(domain, **kwargs)


def inspect(domain=None):
    from allocation23.environments import build_environment
    cases = []
    for selected in ([domain] if domain else ('radio', 'v5')):
        env, spec, physical = build_environment(selected, 3)
        try:
            cases.append(dict(domain=selected, chapter=3, objects=spec.groups,
                horizon=spec.horizon, public_observation_dim=spec.groups*spec.local_dim+spec.global_dim,
                comparator_action_dim=spec.groups+1, initial_budget=spec.initial_budget,
                total_mode='adaptive', full_spend_required=True,
                resource_unit=spec.resource_unit, shared_physics='allocation23.environments',
                state='allocation23.state', completion_geometry='allocation23.full_spend'))
        finally:
            env.close()
    return cases
