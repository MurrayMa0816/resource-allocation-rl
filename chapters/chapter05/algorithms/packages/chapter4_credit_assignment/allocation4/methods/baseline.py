"""Device-aware faithful port of the released complete-MC PPO update."""
import numpy as np
import torch
from allocation23.learning import PPO
from ..common import tensor_batch, subset

class MCPPO:
    def __init__(self, spec, event_dim, seed, device):
        self.device = device

    def update(self, actor, critic, actor_optimizer, critic_optimizer, batches, rng):
        d = tensor_batch(batches, self.device)
        s, raw, old, mask = d['states'], d['raw'], d['old_logp'], d['mask']
        active = mask > 0
        if not active.any():
            return dict(actor_loss=0., value_loss=0., macro_kl=0., optimizer_steps=0)
        adv = d['returns'][:,None] - d['values']
        scale = adv[active].std(unbiased=False)
        adv = (adv-adv[active].mean())/(scale+1e-8)
        with torch.no_grad():
            error = float(((actor.log_prob(s, raw)-old)*mask).abs().max())
        if error > 1e-7:
            raise AssertionError(('likelihood mismatch', error))
        stats = []
        stop = False
        for epoch in range(PPO['epochs']):
            indices = rng.permutation(len(raw))
            for offset in range(0, len(raw), PPO['minibatch']):
                ix = torch.as_tensor(indices[offset:offset+PPO['minibatch']], device=self.device)
                st, m = subset(s, ix), mask[ix]
                lp = actor.log_prob(st, raw[ix])
                delta = lp-old[ix]
                ratio = delta.clamp(-40,40).exp()
                loss = -(torch.minimum(ratio*adv[ix], ratio.clamp(1-PPO['clip'],1+PPO['clip'])*adv[ix])*m).sum(-1).mean()
                actor_optimizer.zero_grad()
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(actor.parameters(), PPO['grad_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite actor gradient')
                actor_optimizer.step()
                prediction = critic(st, raw[ix])
                vl = (((prediction-d['returns'][ix,None])**2)*m).sum()/m.sum().clamp_min(1)
                critic_optimizer.zero_grad()
                vl.backward()
                norm=torch.nn.utils.clip_grad_norm_(critic.parameters(), PPO['grad_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite critic gradient')
                critic_optimizer.step()
                with torch.no_grad():
                    kl = ((ratio-1-delta)*m).sum(-1).mean()
                stats.append([float(loss),float(vl),float(kl)])
                if float(kl) > PPO['macro_kl_stop']:
                    stop = True
                    break
            if stop:
                break
        a = np.asarray(stats)
        return dict(actor_loss=float(a[:,0].mean()), value_loss=float(a[:,1].mean()),
                    macro_kl=float(a[:,2].mean()), optimizer_steps=len(a),
                    advantage_std=float(scale), likelihood_error=error)

    def state_dict(self):
        return {}

    def metadata(self):
        return dict(backend='released MC-PPO device port',ppo=PPO,
                    terminal_only=True,gamma=1.,auxiliary_models=False)

    def load_state_dict(self, state):
        if state:
            raise ValueError('MC-PPO has no auxiliary state')
