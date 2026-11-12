# SPDX-License-Identifier: Apache-2.0
"""Lossless wrapper around the unchanged native8-iteration Torch update.

Only the steady region is graphable: Python-double Adam corrections are exactly
one and no pre-increment recycle predicate is true. No tensor math is rewritten.
"""
from __future__ import annotations
import math
import torch


def eligible(agent, iterations=8):
    start=agent.n_updates
    return (agent.device.type=='cuda' and iterations==8 and agent.rr==8 and start>=40000
            and not any(k>0 and k%agent.recycle_period==0 for k in range(start,start+iterations)))


def _optimizers(agent):
    return (agent.actor_optimizer,agent.critic_optimizer,agent.alpha_optimizer)


def _parameters(agent):
    return list(agent.actor.parameters())+list(agent.critic.parameters())+list(agent.critic_target.parameters())+[agent.log_alpha]


def _snapshot(agent):
    ps=_parameters(agent)
    return {'values':[p.detach().clone() for p in ps],
            'gradients':[(p.grad,None if p.grad is None else p.grad.detach().clone()) for p in ps],
            'slots':[{p:{k:v.clone() for k,v in opt.state[p].items()} for p in opt.parameters}
                     for opt in _optimizers(agent)],
            'iterations':[opt.iterations for opt in _optimizers(agent)],
            'updates':agent.n_updates,'losses':agent._last_losses,
            'CPU_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state(agent.device)}


@torch.no_grad()
def _restore(agent,state):
    for p,value,(grad,savedgrad) in zip(_parameters(agent),state['values'],state['gradients']):
        p.copy_(value);p.grad=grad
        if grad is not None:grad.copy_(savedgrad)
    for opt,slots,count in zip(_optimizers(agent),state['slots'],state['iterations']):
        for p,values in slots.items():
            for key,value in values.items():opt.state[p][key].copy_(value)
        opt.iterations=count
    agent.n_updates=state['updates'];agent._last_losses=state['losses']
    torch.set_rng_state(state['CPU_RNG']);torch.cuda.set_rng_state(state['CUDA_RNG'],agent.device)


class SteadyEightGraph:
    def __init__(self,agent,batches):
        if not eligible(agent,len(batches)):
            raise ValueError('early or event-crossing native updates must stay eager')
        self.agent=agent;self.replays=0;self.rolled_back_warmup_iterations=0
        for opt in _optimizers(agent):
            if opt.iterations!=agent.n_updates or (opt.beta1,opt.beta2)!=(.9,.999):
                raise RuntimeError('steady graph requires synchronized unchanged TF Adam clocks/betas')
            if opt.lr*math.sqrt(1-opt.beta2**(opt.iterations+1))/(1-opt.beta1**(opt.iterations+1))!=opt.lr:
                raise RuntimeError('Adam correction has not exactly rounded to one')
        self.static_batches=torch.empty_like(batches);self.static_batches.copy_(batches)
        state=_snapshot(agent)
        current=torch.cuda.current_stream(agent.device)
        stream=torch.cuda.Stream(device=agent.device);stream.wait_stream(current)
        # One side-stream warmup initializes cuBLAS/autograd state. Every
        # learning tensor, slot, clock, gradient and RNG is then rolled back.
        with torch.cuda.stream(stream):
            for batch in self.static_batches:agent._update_batch(batch)
        stream.synchronize();self.rolled_back_warmup_iterations=8
        _restore(agent,state)
        stream.wait_stream(current)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            for batch in self.static_batches:agent._update_batch(batch)
        self.captured_losses=agent._last_losses
        self.captured_gradients=[p.grad for p in _parameters(agent)]
        # Capture executes Python bookkeeping. Restoring it is mandatory; the
        # captured CUDA work is launched only by graph.replay below.
        _restore(agent,state)
        current.wait_stream(stream)
        self.capture_traced_native_iterations=8

    def replay(self,batches):
        a=self.agent
        if not eligible(a,len(batches)) or batches.shape!=self.static_batches.shape:
            raise ValueError('graph cannot execute this native clock/shape')
        if any(opt.iterations!=a.n_updates for opt in _optimizers(a)):
            raise RuntimeError('optimizer/agent clocks diverged')
        self.static_batches.copy_(batches)
        self.graph.replay()
        a.n_updates+=8
        for opt in _optimizers(a):opt.iterations+=8
        for p,grad in zip(_parameters(a),self.captured_gradients):p.grad=grad
        a._last_losses=self.captured_losses
        self.replays+=1
