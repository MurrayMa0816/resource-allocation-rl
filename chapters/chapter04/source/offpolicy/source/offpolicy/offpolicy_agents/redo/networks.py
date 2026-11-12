# SPDX-License-Identifier: Apache-2.0
"""Explicit Torch port of the ReDo TF-Agents state-SAC networks/Adam math."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class DenseMLP(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(inputs, 256), nn.Linear(256, 256), nn.Linear(256, outputs)])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, capture=False):
        activations = []
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            if capture:
                activations.append(x.detach())
        out = self.layers[-1](x)
        return (out, activations) if capture else out


class SquashedActor(DenseMLP):
    def __init__(self, observations, actions):
        super().__init__(observations, 2*actions)
        self.action_dim = actions

    def sample(self, obs, noise=None, capture=False, deterministic=False):
        output = self.forward(obs, capture=capture)
        params, activations = output if capture else (output, [])
        mean, logstd = params.split(self.action_dim, dim=-1)
        # The checked-in RigL/TF-Agents projection uses exp directly, without
        # SB3's logstd clamp. Do not silently add a different distribution.
        if deterministic:
            raw = mean
        else:
            eps = torch.randn_like(mean) if noise is None else noise
            raw = mean + torch.exp(logstd)*eps
        action = torch.tanh(raw)
        logp = (-.5*((raw-mean)*torch.exp(-logstd)).square()-logstd-.5*math.log(2*math.pi))
        logp = (logp - 2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1)
        return action, logp, activations


class TwinCritic(nn.Module):
    def __init__(self, observations, actions):
        super().__init__()
        self.q1 = DenseMLP(observations+actions, 1)
        self.q2 = DenseMLP(observations+actions, 1)

    def forward(self, obs, action, capture=False):
        x = torch.cat([obs, action], -1)
        if capture:
            q1, a1 = self.q1(x, capture=True)
            q2, a2 = self.q2(x, capture=True)
            return q1.squeeze(-1), q2.squeeze(-1), [a1, a2]
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class DenseTFAdam:
    """Keras Adam epsilon-hat and shared iteration; no per-weight step reset.

    The full denominator is sqrt(v)+epsilon_hat, before bias correction.
    Initializing slots makes update/recycle behavior explicit and avoids a
    hidden TensorFlow-versus-Torch epsilon-position change.
    """
    def __init__(self, parameters, lr=3e-4, epsilon=1e-7, beta1=.9, beta2=.999):
        self.parameters = list(parameters)
        self.lr, self.epsilon = float(lr), float(epsilon)
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.iterations = 0
        self.state = {p: {'m':torch.zeros_like(p), 'v':torch.zeros_like(p)} for p in self.parameters}

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.iterations += 1
        rate = self.lr*math.sqrt(1-self.beta2**self.iterations)/(1-self.beta1**self.iterations)
        grads = [p.grad for p in self.parameters]
        if any(g is None or g.is_sparse for g in grads):
            raise RuntimeError('DenseTFAdam requires every registered parameter gradient')
        ms = [self.state[p]['m'] for p in self.parameters]
        vs = [self.state[p]['v'] for p in self.parameters]
        torch._foreach_mul_(ms, self.beta1)
        torch._foreach_add_(ms, grads, alpha=1-self.beta1)
        torch._foreach_mul_(vs, self.beta2)
        torch._foreach_addcmul_(vs, grads, grads, value=1-self.beta2)
        denominators = torch._foreach_sqrt(vs)
        torch._foreach_add_(denominators, self.epsilon)
        torch._foreach_addcdiv_(self.parameters, ms, denominators, value=-rate)

    @torch.no_grad()
    def clear_kernel_mask(self, parameter, mask):
        self.state[parameter]['m'].masked_fill_(mask, 0)
        self.state[parameter]['v'].masked_fill_(mask, 0)
