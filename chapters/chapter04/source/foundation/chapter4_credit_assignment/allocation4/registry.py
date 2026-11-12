"""Registered methods; names do not imply original-paper reproduction."""
METHODS = ('mc_ppo', 'mc_pg', 'cv_mse', 'lax', 'cocoa', 'cca_state', 'cca', 'gae', 'rudder')

def make_method(name, spec, event_dim, seed, device):
    if name == 'mc_ppo':
        from .methods.baseline import MCPPO
        return MCPPO(spec, event_dim, seed, device)
    if name in ('mc_pg', 'cv_mse', 'lax'):
        from .methods.lax import Method
        return Method(spec, event_dim, seed, device, mode=name)
    if name in ('cocoa', 'cca', 'cca_state'):
        from .methods.hindsight import CocoaMethod, CCAMethod
        if name == 'cocoa':
            return CocoaMethod(spec, event_dim, seed, device)
        return CCAMethod(spec, event_dim, seed, device, mode=name)
    if name in ('gae', 'rudder'):
        from .methods.redistribution import RedistributionMethod
        return RedistributionMethod(spec, event_dim, seed, device, mode=name)
    raise ValueError(name)
