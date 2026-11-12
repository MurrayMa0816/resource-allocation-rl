"""C-PPO collection facade, extracted from the screened native online runtime.

Only the C-PPO path is retained. The original safety-state, finite-horizon
normalizer, counters and endpoint serialization remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from ._c_ppo_core import Agent, Config


def make(method, domain, *, seed=0, device='cpu', profile='formal', overrides=None):
    if method != 'c_ccpo' or domain not in ('v5', 'radio'):
        raise ValueError('This private facade only constructs the C-PPO adapter')
    if profile != 'formal':
        raise ValueError('Override explicit parameters for engineering gates')
    d, a, horizon = (242, 121, 24) if domain == 'v5' else (303, 101, 32)
    params = dict(overrides or {})
    if 'device' in params and params['device'] != device:
        raise ValueError('Configuration and explicit execution device disagree')
    params.pop('device', None)
    if 'horizon' in params and params['horizon'] != horizon:
        raise ValueError('Native physical horizon is fixed')
    params.pop('horizon', None)
    expected = 1.5 - 0.5 / horizon
    if 'weighted_cost_normalizer' in params and params['weighted_cost_normalizer'] != expected:
        raise ValueError('Full-spend SPF cost normalizer must use its analytic bound')
    params['weighted_cost_normalizer'] = expected
    if 'hidden_sizes' in params:
        params['hidden_sizes'] = tuple(params['hidden_sizes'])
    agent = Agent(d, a, Config(**dict(params, device=device)), seed)
    return OnlineAgent(domain, agent, seed, device, profile)


class OnlineAgent:
    def __init__(self, domain, agent, seed, device, profile):
        self.method = 'c_ccpo'
        self.domain, self.agent = domain, agent
        self.seed, self.device, self.profile = int(seed), device, profile
        self.environment_steps = self.episodes = self.condition_index = 0
        self.evaluation_only = False

    @property
    def metadata(self):
        meta = deepcopy(self.agent.metadata())
        for key in ('candidate_only', 'active_registry_changed', 'active_route_modified'):
            meta.pop(key, None)
        return dict(method=self.method, domain=self.domain, seed=self.seed, device=self.device,
                    profile=self.profile, online_from_scratch=True, offline_data=False,
                    environment_steps=self.environment_steps, episodes=self.episodes,
                    condition_index=self.condition_index, algorithm=meta)

    def act(self, observation, history_cost=0., deterministic=False):
        return self.agent.act(observation, history_cost, deterministic,
                              record=not deterministic)

    def observe(self, obs, action, reward, cost, nxt, done, *, history_cost=0., episode_step=0):
        if self.evaluation_only:
            raise ValueError('This fixed terminal model is evaluation-only')
        result = {'learning_reward': self.agent.observe(obs, action, reward, cost, nxt, done)}
        self.environment_steps += 1
        if done:
            self.episodes += 1
        return result

    def finish(self):
        return self.agent.update(terminal=True)

    def save(self, directory):
        import torch
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        payload = directory / 'state.pt'
        torch.save(self.agent.state_dict(full=False), payload)
        (directory / 'online_manifest.json').write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return payload

    @classmethod
    def load(cls, directory, *, expected_domain=None, device='cpu'):
        """Load the fixed inference endpoint on an explicit execution device.

        The source metadata is checked before translating only its device field
        in RAM. This supports the original CUDA endpoints on a CPU host without
        rewriting their files or pretending they contain resumable training state.
        """
        import torch
        directory = Path(directory)
        meta = json.loads((directory / 'online_manifest.json').read_text())
        if meta['method'] != 'c_ccpo' or meta['device'] not in ('cpu', 'cuda'):
            raise ValueError('Unexpected C-PPO endpoint identity')
        if expected_domain is not None and meta['domain'] != expected_domain:
            raise ValueError('Endpoint domain differs')
        state = torch.load(directory / 'state.pt', map_location=device, weights_only=False)
        if state.get('evaluation_only') is not True or 'learning_state' in state:
            raise ValueError('Only fixed evaluation endpoints can be loaded')
        if json.loads(json.dumps(state['metadata'])) != meta['algorithm']:
            raise ValueError('Endpoint payload and manifest metadata differ')
        if state['metadata']['config']['device'] != meta['device']:
            raise ValueError('Endpoint payload and manifest devices differ')
        params = dict(meta['algorithm']['config'])
        params.pop('device')
        result = make('c_ccpo', meta['domain'], seed=meta['seed'], device=device,
                      profile='formal', overrides=params)
        translated = deepcopy(state)
        translated['metadata']['config']['device'] = device
        result.agent.load_state_dict(translated, full=False)
        result.profile = meta['profile']
        result.environment_steps = int(meta['environment_steps'])
        result.episodes = int(meta['episodes'])
        result.condition_index = int(meta['condition_index'])
        result.evaluation_only = True
        return result
