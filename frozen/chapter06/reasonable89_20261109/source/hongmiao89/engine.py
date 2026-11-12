"""89-row scenario hook around byte-unchanged physical algorithms."""
import hashlib
import json
from irrigation_ch6._core.hongmiao import Hongmiao93Env
from irrigation_ch6._core.direct_env import DirectIrrigationEnv
from .scenario import load_hongmiao89_scenario
from .compact_observation import (compact_current_report, compact_policy_observation,
                                  make_compact_observation_space)
from .constants import CASE_ID, EXCLUDED_IDS, BUDGET_FORMULA, DEFAULT_BUDGET_M3


class Hongmiao89Engine(Hongmiao93Env):
    chapter_id = 4
    research_problem = 'reduced89_intertemporal_irrigation'

    def _load_case_scenario(self, **kwargs):
        return load_hongmiao89_scenario(**kwargs)

    def __init__(self, *, daily_quota_m3=DEFAULT_BUDGET_M3 / 81, **kwargs):
        super().__init__(daily_quota_m3=daily_quota_m3, **kwargs)
        self.observation_space = make_compact_observation_space(self.chapter)
        if self.N != 89 or self._storage.shape != (89,) or self._ledger.shape != (89, 1):
            raise ArithmeticError('Physical engine did not instantiate exactly 89 objects')

    def _scenario_fingerprint(self):
        payload = dict(base=DirectIrrigationEnv._scenario_fingerprint(self),
                       case=CASE_ID, calendar=self._calendar,
                       excluded_ids=EXCLUDED_IDS, research_problem=self.research_problem)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _observation(self):
        if not self._initialized:
            raise RuntimeError('Reset before reading the compact observation')
        return compact_current_report(
            self._history[self._day], day_index=self._day,
            available_water_m3=self._allowance(), rice_area_ha=self._areas,
            field_parameters=self._field_parameters, previous_weather=self._previous_weather,
            start_date=self.scenario.start_date, calendar=self._calendar,
            yield_lambdas=self._case_model['jensen']['lambda'])

    def policy_observation(self, public_observation):
        return compact_policy_observation(
            public_observation, chapter=self.chapter,
            start_date=self.scenario.start_date, calendar=self._calendar,
            yield_lambdas=self._case_model['jensen']['lambda'])

    def _case_info(self, info):
        result = super()._case_info(info)
        result.update(environment_version=CASE_ID,
                      budget_provenance='reduced-case explicit budget; default '+BUDGET_FORMULA)
        return result

    def model_config(self):
        result = super().model_config()
        result.update(case_id=CASE_ID, n_regions=89, parent_case_id='Hongmiao93-v2',
                      excluded_region_ids=list(EXCLUDED_IDS),
                      case_status='reduced research subset; not a complete published irrigation district',
                      budget_provenance='reduced-case explicit budget; default '+BUDGET_FORMULA)
        return result
