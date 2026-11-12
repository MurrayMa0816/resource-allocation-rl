"""Chapter tasks around one Hongmiao water, weather and yield implementation.

These classes implement environments, not the algorithms proposed for them.
All four classes directly inherit the common case implementation, so a
chapter-specific extension cannot silently change another chapter's task.
Chapter 5 intentionally shares chapter 4's task: sparse-reward learning is an
algorithmic problem, and its study must not change the physical objective.
"""
from __future__ import annotations

import numpy as np

from .hongmiao import Hongmiao93Env
from .regional_guarantee import RegionalGuarantee


class Chapter3Env(Hongmiao93Env):
    """High-dimensional allocation of an externally fixed actual daily total."""

    chapter_id = 3
    research_problem = "high_dimensional_allocation"


class Chapter4Env(Hongmiao93Env):
    """Allocation with a finite seasonal budget that can be retained over time."""

    chapter_id = 4
    research_problem = "intertemporal_resource_tradeoff"


class Chapter5Env(Hongmiao93Env):
    """The same carryover task, for studying terminal-reward credit assignment."""

    chapter_id = 5
    research_problem = "sparse_terminal_reward_learning"


class Chapter6Env(Hongmiao93Env):
    """Carryover allocation with expected terminal shortfall limits per region.

    Targets and tolerances are required experimental inputs, not inferred
    management standards. They are fixed for an environment instance. The
    reward and physical action constraints remain those of chapters 4 and 5.
    A cost sample is emitted at step 81; feasibility concerns its expectation
    over complete episodes, not a single episode's outcome.
    """

    chapter_id = 6
    research_problem = "regional_long_term_guarantees"

    def __init__(self, *, guarantee_target, shortfall_tolerance, **kwargs):
        self._guarantee = RegionalGuarantee(
            guarantee_target=guarantee_target,
            shortfall_tolerance=shortfall_tolerance, n_regions=93)
        super().__init__(**kwargs)

    def constraint_spec(self):
        """Detached, fixed constraint definition in published region order."""
        result = self._guarantee.specification()
        result["region_ids"] = list(self.scenario.region_ids)
        return result

    def constraint_feedback(self):
        """One episode's cost sample, never a declaration of expected feasibility.

        Before terminal evaluation, zeros are placeholders with an explicit
        false flag. This method reconstructs feedback from public outcomes;
        no separate mutable cost history or checkpoint cache is introduced.
        """
        if not self._initialized:
            raise RuntimeError("reset must be called before constraint_feedback")
        evaluated = self._day == self.T
        costs = np.zeros(self.N, dtype=np.float64)
        if evaluated:
            public = self.public_observation()
            if not np.all(public["settled_mask"]):
                raise ArithmeticError("regional constraints require all terminal outcomes")
            costs = self._guarantee.costs(public["settled_relative_yield"])
        return {"constraint_costs": costs,
                "constraints_evaluated": evaluated}

    def summarize_guarantees(self, relative_yield_samples):
        """Evaluate completed episodes in the published region order.

        The caller must use the same policy, constraint definition and stated
        weather protocol. Repeating a fixed historical trajectory does not
        create independent weather samples. Outputs are empirical estimates,
        not certified policy guarantees.
        """
        result = self._guarantee.summarize(relative_yield_samples)
        result["region_ids"] = list(self.scenario.region_ids)
        return result

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        info.update(self.constraint_feedback())
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(self.constraint_feedback())
        return observation, reward, terminated, truncated, info

    def episode_summary(self):
        result = super().episode_summary()
        if self._initialized:
            feedback = self.constraint_feedback()
            result["constraint_feedback"] = {
                "constraint_costs": feedback["constraint_costs"].tolist(),
                "constraints_evaluated": feedback["constraints_evaluated"],
            }
        return result

    def render_context(self):
        result = super().render_context()
        result["regional_guarantee"] = self.constraint_spec()
        feedback = self.constraint_feedback()
        result["constraint_feedback"] = {
            "constraint_costs": feedback["constraint_costs"].tolist(),
            "constraints_evaluated": feedback["constraints_evaluated"],
        }
        return result
