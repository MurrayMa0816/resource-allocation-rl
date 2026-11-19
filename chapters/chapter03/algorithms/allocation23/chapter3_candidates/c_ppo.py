"""C-PPO task adapter: native algorithm, public-state accounting, actor-only factor.

Online learning starts from scratch. The main-project integration preserves
the screened task profile; it does not establish a new formal benchmark result.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from ._c_ppo_runtime import make, OnlineAgent


class Candidate:
    def __init__(self, obs_dim, action_dim, seed, device='cpu', config=None):
        if device not in ('cpu', 'cuda'):
            raise ValueError('C-PPO device must be cpu or cuda')
        self.domain = {(303,101):'radio', (242,121):'v5'}[(int(obs_dim),int(action_dim))]
        self.obs_dim, self.action_dim, self.seed = int(obs_dim), int(action_dim), int(seed)
        self.horizon = 32 if self.domain == 'radio' else 24
        params = dict(config or {})
        if 'reward_scale' in params and params.pop('reward_scale') != 1.:
            raise ValueError('Native C-PPO uses reward fractions with advantage normalization')
        if 'horizon' in params and params.pop('horizon') != self.horizon:
            raise ValueError('Native horizon cannot change')
        self.online = make('c_ccpo', self.domain, seed=self.seed, device=device,
                           profile='formal', overrides=params)
        self.config = asdict(self.online.agent.config)
        self.evaluation_only = False

    def act(self, obs, deterministic=False):
        obs = np.asarray(obs, np.float32)
        if not deterministic and self.evaluation_only:
            raise RuntimeError('Loaded reference model is evaluation only')
        return self.online.act(obs, history_cost=1.-float(obs[-1]),
                               deterministic=bool(deterministic))

    def observe(self, obs, action, native_reward, next_obs, done):
        if self.evaluation_only:
            raise RuntimeError('Loaded reference model forbids online observe')
        obs, nxt = np.asarray(obs,np.float32), np.asarray(next_obs,np.float32)
        step = self.online.environment_steps % self.horizon
        if bool(done) != (step == self.horizon-1):
            raise ValueError('Reference transition must complete the native horizon')
        # Only public float32 balance features are read. The collector separately
        # verifies the native exact physical ledger, full spend and action replay.
        cost = float(obs[-1])-float(nxt[-1])
        before = self.online.agent.updates
        result = self.online.observe(obs, action, float(native_reward), cost, nxt, bool(done),
                                     history_cost=1.-float(obs[-1]), episode_step=step)
        metrics = dict(result)
        if self.online.agent.updates != before:
            metrics.update(self.online.agent.last_metrics)
        return metrics

    def finish(self):
        return self.online.finish()

    def accounting(self):
        a = self.online.agent
        return dict(environment_steps=self.online.environment_steps,
                    complete_episodes=self.online.episodes, optimizer_blocks=a.updates,
                    policy_updates=a.policy_updates, value_updates=a.value_updates,
                    early_stops=a.early_stops, pending_behavior_action=int(a._pending is not None),
                    rollout_transitions=len(a._buffer))

    def save(self, directory):
        return self.online.save(directory)

    @classmethod
    def load(cls, directory, device='cpu'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('C-PPO device must be cpu or cuda')
        meta = json.loads((Path(directory)/'online_manifest.json').read_text())
        if meta['method'] != 'c_ccpo' or meta['device'] not in ('cpu', 'cuda'):
            raise ValueError('Unexpected reference checkpoint identity')
        result = cls.__new__(cls)
        result.online = OnlineAgent.load(directory, expected_domain=meta['domain'], device=device)
        result.domain, result.seed = meta['domain'], int(meta['seed'])
        result.obs_dim, result.action_dim = ((303,101) if result.domain == 'radio' else (242,121))
        result.horizon = 32 if result.domain == 'radio' else 24
        result.config = asdict(result.online.agent.config)
        result.evaluation_only = True
        return result
