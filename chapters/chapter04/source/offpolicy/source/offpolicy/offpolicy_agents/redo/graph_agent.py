# SPDX-License-Identifier: Apache-2.0
"""Optional execution wrapper; agent.py remains the immutable eager reference."""
from __future__ import annotations
import math
import numpy as np
import torch
try:
    from .agent import Agent as EagerAgent
    from .steady_graph import SteadyEightGraph,eligible
except ImportError:
    from agent import Agent as EagerAgent
    from steady_graph import SteadyEightGraph,eligible


class Agent(EagerAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._steady_graph=None;self._graph_blocks=0;self._eager_blocks=0
        self._last_execution='not_started'

    def update_block(self,batches):
        if eligible(self,len(batches)):
            if self._steady_graph is None:self._steady_graph=SteadyEightGraph(self,batches)
            self._steady_graph.replay(batches);self._graph_blocks+=1
            self._last_execution='steady_graph'
        else:
            for batch in batches:self._update_batch(batch)
            self._eager_blocks+=1;self._last_execution='eager_early_or_event'

    def observe(self,obs,action,reward,next_obs,done):
        # Same validation, paid transition, sampling RNG and one upload as the
        # reference observe. Only dispatch of the eight existing updates differs.
        if self._inference_only:raise RuntimeError('inference endpoint is not a continuation checkpoint')
        o=self._observation(obs);no=self._observation(next_obs);a=np.asarray(action,dtype=np.float32)
        if a.shape!=(self.action_dim,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.000001 or not math.isfinite(reward):
            raise ValueError('invalid native transition')
        self.replay.add(o,a,float(reward),no,bool(done));self.steps+=1
        before=len(self.events)
        if self.steps>self.warmup:
            idx=self.rng.integers(0,self.replay.size,size=(self.rr,self.batch_size))
            batches=torch.as_tensor(self.replay.data[idx],device=self.device)
            self.update_block(batches)
        return {'steps':self.steps,'updates':self.n_updates,'recycle_events':self.events[before:]}

    def diagnostics(self):
        result=super().diagnostics()
        result['execution']={'last_path':self._last_execution,'graph_blocks':self._graph_blocks,
            'eager_blocks':self._eager_blocks,'steady_threshold':40000,
            'rolled_back_graph_warmup_iterations':0 if self._steady_graph is None else self._steady_graph.rolled_back_warmup_iterations,
            'graph_capture_traced_iterations':0 if self._steady_graph is None else self._steady_graph.capture_traced_native_iterations,
            'tensor_update_math_rewritten':False}
        return result
