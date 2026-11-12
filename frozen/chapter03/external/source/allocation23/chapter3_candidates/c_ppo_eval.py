"""Compatibility loader for historical C-PPO inference endpoints.

Training uses c_ppo.Candidate. This name remains explicitly evaluation-only so
that a former deployment adapter can never silently become a training class.
"""
from .c_ppo import Candidate as _TrainingCandidate


class Candidate(_TrainingCandidate):
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Use c_ppo.Candidate for online training')

    def save(self, directory):
        raise RuntimeError('Evaluation compatibility adapter cannot rewrite endpoints')
