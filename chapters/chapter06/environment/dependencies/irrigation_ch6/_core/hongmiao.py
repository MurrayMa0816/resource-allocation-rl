"""Hongmiao93 research environment: 81 daily decisions and terminal yield projection.

The 93 agricultural areas are published unit data. Crop/soil parameters are
same-district transfers. Field-capacity initial water and frozen percolation values
are research distributions, not measurements. Import and execution use
packaged inputs, never the network or fresh parameter sampling.
"""
from __future__ import annotations
import copy,csv,hashlib,io,json,math
from datetime import date,timedelta
from importlib import resources
from numbers import Integral,Real
import numpy as np
from gymnasium import spaces
from .compact_observation import (compact_current_report, compact_policy_observation,
                                 make_compact_observation_space)
from .direct_env import DirectIrrigationEnv
from .direct_execution import FIELD_PARAMETER_NAMES
from .direct_scenarios import DirectScenario
from .field import FieldParams
from .weather import WeatherDay,PARAM_NAMES

CASE_ID='Hongmiao93-v2'
ACTIVE_DAYS=81
PHYSICAL_DAYS=81
CALENDAR_MATURITY_DAYS=97
OUTCOME_NAMES=('tillering','booting','heading','milky')
PARAMETER_GENERATION_SEED=20261013
STORAGE_GENERATION_SEED=20261014
REFERENCE_ROOT_AVAILABLE_WATER_MM=48.0
DEFAULT_STORAGE_VARIATION=.25
DEFAULT_RAIN_MULTIPLIER=0.0
DEFAULT_WEATHER_MODE='persistent_drought'
INITIAL_THETA=.39
DEFAULT_DAILY_QUOTA_M3=10000.0
FROZEN_PARAMETER_TABLE='region_parameters_frozen.csv'
FIXED_FIELD_PARAMETERS={
    'theta_sat':.5,'theta_fc':.39,'theta_wp':.23,
    'theta_crit':.4,'h_max_mm':80.0,'drain_power':1.0,'evap_power':1.0,
}
PARAMETER_IDENTITY='research_distribution_not_regional_measurements'
BUDGET_PROVENANCE=(
    'research_setting: fixed 10000 m3/day in the default persistent drought case; '
    'selected below a conservative all-policy current-stress-refill bound, not by '
    'reward or a demand ratio; no runtime demand scaling; not measured supply'
)

def _inputs():
    directory=resources.files('irrigation_ch6._core').joinpath('data','hongmiao93')
    manifest=json.loads(directory.joinpath('manifest.json').read_text())
    if manifest.get('schema')!='hongmiao93_packaged_v4':
        raise ValueError('unsupported Hongmiao packaged-input schema')
    required={'regions_93.csv',FROZEN_PARAMETER_TABLE,'weather.csv',
              'weather_prior.json','quota.csv','budget_contract.json',
              'case_model.json','source_registry.csv'}
    if (manifest.get('case_id')!=CASE_ID or manifest.get('n')!=93
            or manifest.get('physical_days')!=ACTIVE_DAYS
            or manifest.get('decision_days')!=ACTIVE_DAYS
            or manifest.get('parameter_generation_seed')!=PARAMETER_GENERATION_SEED
            or manifest.get('parameter_table')!=FROZEN_PARAMETER_TABLE
            or manifest.get('fixed_field_parameters')!=FIXED_FIELD_PARAMETERS
            or manifest.get('initial_theta')!=INITIAL_THETA
            or manifest.get('parameter_generation',{}).get('seed')!=PARAMETER_GENERATION_SEED
            or manifest.get('parameter_generation',{}).get('identity')!=PARAMETER_IDENTITY
            or manifest.get('storage_generation_seed')!=STORAGE_GENERATION_SEED
            or not required.issubset(manifest.get('files_sha256',{}))):
        raise ValueError('Hongmiao packaged manifest is incomplete or stale')
    for name,digest in manifest['files_sha256'].items():
        if '/' in name or '\\' in name or name in ('.','..'):
            raise ValueError('invalid packaged input filename')
        if hashlib.sha256(directory.joinpath(name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Hongmiao input checksum mismatch: {name}')
    def table(name):
        return list(csv.DictReader(io.StringIO(directory.joinpath(name).read_text(encoding='utf-8-sig'))))
    return directory,manifest,table

def load_hongmiao_scenario(*,profile='heterogeneous',historical_year=2022,
                          daily_quota_m3=DEFAULT_DAILY_QUOTA_M3,
                          storage_variation=DEFAULT_STORAGE_VARIATION):
    """Load one frozen parameter table and a constant 81-day water setting.

    ``uniform_percolation`` changes only percolation to 2 mm/day. Its initial
    state, soil, area, weather and calendar match the heterogeneous case.
    Actual daily totals are fixed in chapter 3; chapters 4--6 share their seasonal sum.
    """
    if profile not in ('heterogeneous','uniform_percolation'):
        raise ValueError("profile must be 'heterogeneous' or 'uniform_percolation'")
    if (isinstance(storage_variation,(bool,np.bool_))
            or not isinstance(storage_variation,Real)):
        raise ValueError('storage_variation must be finite and in [0, 0.5]')
    try:
        storage_variation=float(storage_variation)
    except (OverflowError,ValueError) as exc:
        raise ValueError('storage_variation must be finite and in [0, 0.5]') from exc
    if not math.isfinite(storage_variation) or not 0. <= storage_variation <= .5:
        raise ValueError('storage_variation must be finite and in [0, 0.5]')
    if isinstance(historical_year,bool) or not isinstance(historical_year,Integral) or historical_year not in (2022,2024):
        raise ValueError('Hongmiao historical_year must be2022 or2024')
    if isinstance(daily_quota_m3,(bool,np.bool_)) or not isinstance(daily_quota_m3,Real):
        raise ValueError('daily_quota_m3 must be finite and nonnegative with a finite 81-day total')
    try:
        daily_quota_m3=float(daily_quota_m3)
    except (OverflowError,ValueError) as exc:
        raise ValueError('daily_quota_m3 must have a finite 81-day total') from exc
    if (not math.isfinite(daily_quota_m3) or daily_quota_m3<0
            or not math.isfinite(daily_quota_m3*ACTIVE_DAYS)):
        raise ValueError('daily_quota_m3 must be finite and nonnegative with a finite 81-day total')
    directory,manifest,table=_inputs()
    model=json.loads(directory.joinpath('case_model.json').read_text())
    if (model.get('schema')!='hongmiao93_model_v4'
            or any(model.get('parameters',{}).get(k,{}).get('value')!=v
                   for k,v in FIXED_FIELD_PARAMETERS.items())):
        raise ValueError('Hongmiao case model is stale')
    rows=table(FROZEN_PARAMETER_TABLE); published=table('regions_93.csv')
    area_by_id={r['gate_id']:float(r['midseason_rice_area_mu'])/15 for r in published}
    if (len(rows)!=93 or len(published)!=93 or len(area_by_id)!=93
            or tuple(r['region_id'] for r in rows)!=tuple(r['gate_id'] for r in published)):
        raise ValueError('Hongmiao93 identity table is incomplete')
    params=[]; initial=[]; initial_surface=[]
    for row in rows:
        area=float(row['area_ha'])
        if not np.isclose(area,area_by_id[row['region_id']],rtol=0,atol=1e-10):
            raise ValueError('region area disagrees with published rice area')
        if (any(float(row[k])!=v for k,v in FIXED_FIELD_PARAMETERS.items())
                or float(row['initial_theta'])!=INITIAL_THETA
                or row['parameter_generation_seed']!=str(PARAMETER_GENERATION_SEED)
                or row['parameter_identity']!=PARAMETER_IDENTITY):
            raise ValueError('Hongmiao frozen soil, initial field capacity or provenance is stale')
        h0=float(row['initial_surface_depth_mm'])
        frozen_percolation=float(row['percolation_sat_mm_day'])
        deviation=float(row['storage_deviation'])
        frozen_s=float(row['root_available_water_mm'])
        if (not math.isfinite(deviation) or abs(deviation)>1.+1e-12
                or not np.isclose(frozen_s,REFERENCE_ROOT_AVAILABLE_WATER_MM*(1.+DEFAULT_STORAGE_VARIATION*deviation),rtol=0,atol=1e-10)
                or not np.isclose(float(row['soil_depth_mm']),frozen_s/(FIXED_FIELD_PARAMETERS['theta_fc']-FIXED_FIELD_PARAMETERS['theta_wp']),rtol=0,atol=1e-10)
                or row['storage_generation_seed']!=str(STORAGE_GENERATION_SEED)):
            raise ValueError('Hongmiao frozen root available storage is inconsistent')
        root_available=REFERENCE_ROOT_AVAILABLE_WATER_MM*(1.+storage_variation*deviation)
        if h0 != 0. or not 1.<=frozen_percolation<=3.:
            raise ValueError('Hongmiao frozen research distribution is outside its declared range')
        p=FieldParams(area_ha=area,kcb=1.,ke_wet=.03,
                      soil_depth_mm=root_available/(FIXED_FIELD_PARAMETERS['theta_fc']-FIXED_FIELD_PARAMETERS['theta_wp']),
                      percolation_sat_mm_day=(2. if profile=='uniform_percolation' else frozen_percolation),
                      **FIXED_FIELD_PARAMETERS)
        params.append(p); initial_surface.append(h0)
        initial.append(p.soil_depth_mm*INITIAL_THETA+h0)
    area=np.array([p.area_ha for p in params])
    if not np.isclose(area.sum(),1327.2,rtol=0,atol=1e-9):
        raise ValueError('Hongmiao published total rice area is inconsistent')
    if not np.isclose(np.average([p.root_available_water_mm for p in params],weights=area),REFERENCE_ROOT_AVAILABLE_WATER_MM,rtol=0,atol=1e-10):
        raise ValueError('Hongmiao root available storage must preserve the area-weighted reference mean')
    records={}
    for row in table('weather.csv'):
        d=date.fromisoformat(row['date']);records[row['date']]=WeatherDay(d,np.array([float(row[k]) for k in PARAM_NAMES]),float(row['ET0_FAO56_mm_day']))
    prior=json.loads(directory.joinpath('weather_prior.json').read_text())
    pools=prior.pop('donor_pools');forecasts={tuple(map(int,k.split(':'))):tuple(v) for k,v in prior.pop('forecast_table_by_year')[str(historical_year)].items()}
    quota_rows=table('quota.csv')
    if len(quota_rows)!=ACTIVE_DAYS or any(
            int(row['physical_day_index'])!=i
            or row['month_day']!=(date(2022,5,30)+timedelta(days=i)).strftime('%m-%d')
            or float(row['daily_quota_m3'])!=DEFAULT_DAILY_QUOTA_M3
            for i,row in enumerate(quota_rows)):
        raise ValueError('Hongmiao packaged daily quota plan is inconsistent')
    budget=json.loads(directory.joinpath('budget_contract.json').read_text())
    if (budget.get('schema')!='hongmiao93_budget_v3'
            or budget.get('default_daily_quota_m3')!=DEFAULT_DAILY_QUOTA_M3
            or budget.get('default_seasonal_budget_m3')!=ACTIVE_DAYS*DEFAULT_DAILY_QUOTA_M3
            or budget.get('decision_days')!=ACTIVE_DAYS):
        raise ValueError('Hongmiao packaged budget contract is inconsistent')
    quota=np.full(ACTIVE_DAYS,daily_quota_m3)
    metadata={
        'case_id':CASE_ID,'parameter_profile':profile,
        'parameter_table':FROZEN_PARAMETER_TABLE,
        'parameter_generation_seed':PARAMETER_GENERATION_SEED,
        'parameter_identity':PARAMETER_IDENTITY,
        'storage_generation_seed':STORAGE_GENERATION_SEED,
        'storage_variation':storage_variation,
        'area_weighted_root_available_water_mm':float(np.average([p.root_available_water_mm for p in params],weights=area)),
        'parameter_generation':copy.deepcopy(manifest['parameter_generation']),
        'initial_soil_state':'field_capacity_theta_0.39_without_ponding',
        'initial_total_water_m3':float(np.dot(area,np.array(initial))*10.),
        'initial_soil_water_m3':float(np.dot(area,[p.soil_depth_mm*INITIAL_THETA for p in params])*10.),
        'initial_surface_water_m3':float(np.dot(area,initial_surface)*10.),
        'area_weighted_initial_surface_depth_mm':float(np.average(initial_surface,weights=area)),
        'area_weighted_percolation_sat_mm_day':float(np.average([p.percolation_sat_mm_day for p in params],weights=area)),
        'daily_quota_m3':daily_quota_m3,
        'seasonal_budget_m3':ACTIVE_DAYS*daily_quota_m3,
        'budget_provenance':BUDGET_PROVENANCE,
        'budget_contract':budget,
        'daily_supply_limit_m3':ACTIVE_DAYS*daily_quota_m3,
        'daily_supply_limit_identity':'seasonal budget as mathematical upper bound; ideal availability, not engineering capacity',
        'regional_delivery_capacity':'not imposed; no calibrated regional engineering capacity',
        'published_rice_area_ha':1327.2,
        'geography':'Hongmiao irrigation service units, Zhanghe district, Hubei; polygon geometries unavailable',
        'outcome':'same-district transferred four-stage Jensen ET response; not measured harvest',
        'evaluation':'terminal model projection at allocation end; no postrun or claim of biological maturity',
        'omitted_ripening':'the four-stage response does not model later ripening stress',
        'sources':copy.deepcopy(manifest['sources']),'manifest':manifest,
    }
    scenario=DirectScenario(region_ids=tuple(r['region_id'] for r in rows),field_params=tuple(params),initial_storage_mm=np.array(initial),heading_dates=tuple(date(historical_year,7,31) for _ in rows),quota_plan=quota,evaluation_mode="terminal_projection",start_date=date(historical_year,5,30),weather_records=records,donor_pools=pools,forecast_table=forecasts,weather_contract=prior,maturity_offset_days=35,field_substeps_per_day=96,metadata=metadata)
    return scenario,model

class Hongmiao93Env(DirectIrrigationEnv):
    """Shared case implementation, instantiated through a chapter subclass.

    Every step advances one day. On the last day the complete four-stage ET
    record is evaluated; September 4 remains a literature-calendar maturity
    reference, not a date that the simulator advances to or observes.
    """
    metadata = {**DirectIrrigationEnv.metadata, "render_modes": ["ansi", "rgb_array", "svg"]}
    chapter_id = None
    research_problem = None
    _requires_all_day_scarcity_certificate = True

    def _load_case_scenario(self, **kwargs):
        """Construct physical parameters before caches, weather and fingerprints.

        Explicitly identified cases may override this hook. The shared v2
        chapters always load the unmodified packaged research parameters.
        """
        return load_hongmiao_scenario(**kwargs)

    def __init__(self,*,profile='heterogeneous',historical_year=2022,
                 daily_quota_m3=DEFAULT_DAILY_QUOTA_M3,
                 storage_variation=DEFAULT_STORAGE_VARIATION,
                 rain_multiplier=DEFAULT_RAIN_MULTIPLIER,
                 weather_mode=DEFAULT_WEATHER_MODE,report_delay_days=0,render_mode=None):
        if self.chapter_id not in (3,4,5,6):
            raise TypeError('Hongmiao93Env is a shared base; use make_env or a Chapter3Env--Chapter6Env class')
        chapter = self.chapter_id
        if report_delay_days is not None and (
                isinstance(report_delay_days,(bool,np.bool_))
                or not isinstance(report_delay_days,Integral) or report_delay_days != 0):
            raise ValueError('all current chapters observe current state; report_delay_days must be 0')
        self.parameter_profile=profile
        self.rain_multiplier=rain_multiplier
        scenario,self._case_model=self._load_case_scenario(profile=profile,historical_year=historical_year,daily_quota_m3=daily_quota_m3,storage_variation=storage_variation)
        self.storage_variation=float(storage_variation)
        self.daily_quota_m3=float(daily_quota_m3)
        self._calendar=self._case_model['calendar']
        self.decision_horizon_steps=ACTIVE_DAYS
        # No engineering delivery limit has been calibrated. This bound only
        # expresses ideal availability of the entire unspent seasonal budget.
        limits=np.full(ACTIVE_DAYS,ACTIVE_DAYS*self.daily_quota_m3)
        requested_render_mode = render_mode
        super().__init__(scenario=scenario,chapter=chapter,daily_limit_m3=limits,report_delay_days=0,local_weather_spread=0.,yield_lambdas=self._case_model['jensen']['lambda'],weather_mode=weather_mode,historical_year=historical_year,render_mode=None if render_mode=='svg' else render_mode)
        self.render_mode = requested_render_mode
        self.observation_space.spaces.update({
            'decision_step_index':spaces.Box(0,self.decision_horizon_steps,shape=(),dtype=np.int64),
            'decision_horizon_steps':spaces.Box(self.decision_horizon_steps,self.decision_horizon_steps,shape=(),dtype=np.int64),
            'remaining_allocation_days':spaces.Box(0,81,shape=(),dtype=np.int64)})
        self.public_observation_space=self.observation_space
        self.observation_space=make_compact_observation_space(self.chapter)
        from .scarcity import scarcity_certificate
        from .weather_pairs import DROUGHT_MIN_ET0_MM
        self._scarcity_certificate = scarcity_certificate(
            field_params=self.scenario.field_params,
            initial_storage_mm=self.scenario.initial_storage_mm,
            crop_kcb=[self._crop_profiles(self.scenario.start_date+timedelta(days=t))[0,1]
                      for t in range(self.T)],
            daily_quota_m3=self.daily_quota_m3,
            min_et0_mm_day=DROUGHT_MIN_ET0_MM if weather_mode == 'persistent_drought' else 0.,
            no_rain=weather_mode == 'persistent_drought' and self.rain_multiplier == 0.,
            chapter=self.chapter, substeps_per_day=self.scenario.field_substeps_per_day)
        if (self._requires_all_day_scarcity_certificate
                and self.chapter == 3 and weather_mode == 'persistent_drought'
                and self.daily_quota_m3 > 0 and not self._scarcity_certificate['certified']):
            raise ValueError('persistent_drought C3 configuration lacks an all-day scarcity certificate; '
                             'use certified parameters or an explicitly noncertified markov/historical control')

    def _make_weather_driver(self, weather_mode, historical_year):
        from .weather_pairs import PairWeatherDriver
        driver=PairWeatherDriver(self.scenario, weather_mode, historical_year,
                                 rain_multiplier=self.rain_multiplier)
        self.rain_multiplier=driver.rain_multiplier
        return driver

    def _scenario_fingerprint(self):
        base=super()._scenario_fingerprint()
        return hashlib.sha256(json.dumps({
            'base':base,'case':CASE_ID,'calendar':self._calendar,
            'chapter':self.chapter_id,'research_problem':self.research_problem,
            'regional_guarantee':self.constraint_spec() if hasattr(self,'constraint_spec') else None,
        },sort_keys=True).encode()).hexdigest()

    def _crop_profiles(self,day):
        md=day.strftime('%m-%d')
        if day>=self.scenario.start_date+timedelta(days=CALENDAR_MATURITY_DAYS):
            return np.tile([5.,0.,0.,-1.],(self.N,1))
        matching=[p for p in self._calendar if p['start_mm_dd']<=md<=p['end_mm_dd_inclusive']]
        if len(matching)!=1:raise ValueError('date absent from Hongmiao crop calendar')
        p=matching[0];oi=p['jensen_index'];stage=(4 if p['stage'].startswith('ripening') else 0) if oi is None else oi
        return np.tile([stage,p['kcb'],p['ke_wet'],-1 if oi is None else oi],(self.N,1))

    def _active_mask(self):
        return super()._active_mask() & (self._day<81)

    def _check_settlement_coverage(self,ledger,selected):
        if np.any(selected):
            if (not np.all(ledger.stage_observations[selected,0,:]==np.array([37,16,10,9])) or not np.all(ledger.reference_et_mm[selected,0,:]>0)):
                raise ArithmeticError('Hongmiao full four-stage ET coverage is incomplete at terminal evaluation')

    def _format_settlements(self,records):
        records=super()._format_settlements(records)
        for r in records:
            r.update(outcome_kind='simulated_relative_water_response_not_observed_harvest',response_scope='complete_four_response_stages_with_ripening_omitted',partial_stage_rule='not_used_full_stage_history_required',maturity_source='River2024_Sep4_reference_not_simulated',evaluation_mode='terminal_projection',evaluated_at_crop_maturity=False,response_stage_names=list(OUTCOME_NAMES))
        return records

    def public_observation(self):
        """Return the complete 26-field public display, using received reports.

        Water and ET are current model reports in every chapter.
        Calendar, budgets and published terminal
        evaluations stay current. The returned arrays never alias live state.
        """
        if not self._initialized:
            raise RuntimeError('reset must be called before public_observation')
        obs=super()._observation()
        obs['decision_step_index']=np.array(min(self._day,ACTIVE_DAYS),dtype=np.int64)
        obs['decision_horizon_steps']=np.array(self.decision_horizon_steps,dtype=np.int64)
        obs['remaining_allocation_days']=np.array(max(0,81-self._day),dtype=np.int64)
        # The base public view creates detached arrays for every field already.
        return obs

    def policy_observation(self,public_observation):
        """Pure conversion of supplied public reports to compact policy input."""
        return compact_policy_observation(
            public_observation, chapter=self.chapter,
            start_date=self.scenario.start_date, calendar=self._calendar,
            yield_lambdas=self._case_model['jensen']['lambda'])

    def _observation(self):
        if not self._initialized:
            raise RuntimeError('reset must be called before public_observation')
        return compact_current_report(
            self._history[self._day], day_index=self._day,
            available_water_m3=self._allowance(), rice_area_ha=self._areas,
            field_parameters=self._field_parameters, previous_weather=self._previous_weather,
            start_date=self.scenario.start_date, calendar=self._calendar,
            yield_lambdas=self._case_model['jensen']['lambda'])

    def water_shortage_diagnostics(self):
        """Current public-state refill deficits; no forecast or physical rollout.

        This is the instantaneous volume to reach Ks=1. It is a necessary
        starting-state refill, not daily full ET, a delivery cap or future ROI.
        A certified configuration keeps its aggregate above Q in C3. Other
        explicitly identified scenarios make no all-day shortage claim.
        """
        if not self._initialized:
            raise RuntimeError('reset must be called before water_shortage_diagnostics')
        storage = self._observation()['reported_field_storage_mm']
        thresholds = np.array([p.soil_depth_mm*p.theta_crit for p in self.scenario.field_params])
        deficit = np.maximum(thresholds-storage, 0.)
        volumes = 10.*self._areas*deficit
        allowance = float(self._allowance())
        return dict(day_index=int(self._day), region_ids=list(self.scenario.region_ids),
                    current_refill_depth_mm=deficit.copy(),
                    current_refill_need_m3=volumes.copy(),
                    total_current_refill_need_m3=float(np.sum(volumes)),
                    available_water_m3=allowance,
                    current_supply_gap_m3=float(np.sum(volumes)-allowance),
                    regions_below_nonstress_threshold=int(np.count_nonzero(deficit > 1e-9)),
                    certified_for_all_decision_days=bool(self._scarcity_certificate['certified']),
                    is_decision_state=self._day < self.T,
                    definition='current refill to root-zone theta_crit; not daily full-ET demand or a hard regional action cap')

    def model_config(self):
        """Detached public fixed configuration, without field truth or RNG state.

        Crop coefficients are dated by ``calendar``. Their initial baseline
        values in ``shared_field_parameters`` reproduce the full public field
        parameter table and are not a replacement for the dated coefficients.
        """
        return copy.deepcopy({
            'case_id':CASE_ID,'chapter':self.chapter,
            'scarcity_certificate':self._scarcity_certificate,
            'shortage_diagnostics_api':'water_shortage_diagnostics',
            'environment_class':type(self).__name__,'research_problem':self.research_problem,
            'regional_guarantee':self.constraint_spec() if hasattr(self,'constraint_spec') else None,
            'parameter_profile':self.parameter_profile,'n_regions':self.N,
            'regional_model':{
                'unit':'whole midseason-rice area within each outlet service range',
                'internal_field_allocation_modeled':False,
                'zero_action':'no irrigation today; stored water, ET and yield history continue evolving',
                'allowed_action_patterns':['sparse','dense'],
                'mandatory_pause_days':0,
                'minimum_positive_allocation_m3':0.,
                'all_regions_must_receive_water':False,
                'sparsity_reward':False,
            },
            'storage_variation':self.storage_variation,
            'rain_multiplier':self.rain_multiplier,
            'root_available_water_definition':'S=Z*(theta_fc-theta_wp); Z in mm; excludes surface ponding; research effective capacity, not measured rooting depth',
            'reference_root_available_water_mm':REFERENCE_ROOT_AVAILABLE_WATER_MM,
            'storage_generation_seed':STORAGE_GENERATION_SEED,
            'region_ids':list(self.scenario.region_ids),
            'shared_field_parameters':{**FIXED_FIELD_PARAMETERS,'kcb':1.,'ke_wet':.03},
            'field_parameter_names':list(FIELD_PARAMETER_NAMES),
            'calendar':self._calendar,
            'yield_lambdas':self._case_model['jensen']['lambda'],
            'outcome_stage_names':list(OUTCOME_NAMES),
            'horizon_days':ACTIVE_DAYS,'decision_horizon_steps':ACTIVE_DAYS,
            'start_date':self.scenario.start_date.isoformat(),
            'heading_date':self.scenario.heading_dates[0].isoformat(),
            'heading_day_index':(self.scenario.heading_dates[0]-self.scenario.start_date).days,
            'maturity_day_index':CALENDAR_MATURITY_DAYS,
            'evaluation_date':(self.scenario.start_date+timedelta(days=ACTIVE_DAYS)).isoformat(),
            'reference_maturity_date':(self.scenario.start_date+timedelta(days=CALENDAR_MATURITY_DAYS)).isoformat(),
            'daily_quota_m3':self.daily_quota_m3,
            'seasonal_budget_m3':ACTIVE_DAYS*self.daily_quota_m3,
            'budget_mode':'fixed_daily_total' if self.chapter==3 else 'carryover_budget',
            'action_sum_constraint':'equal' if self.chapter==3 else 'less_than_or_equal',
            'must_use_every_unit':self.chapter==3,
            'budget_provenance':BUDGET_PROVENANCE,
            'daily_supply_limit_m3':ACTIVE_DAYS*self.daily_quota_m3,
            'daily_supply_limit_identity':'seasonal budget as mathematical upper bound; ideal availability, not engineering capacity',
            'field_substeps_per_day':self.scenario.field_substeps_per_day,
            'evaluation_mode':'terminal_projection',
            'reward_identity':'area_weighted_complete_four_stage_projection_at_allocation_end',
            'gamma':1.,'report_delay_days':int(self._report_delay[0]),
            'weather_mode':self._weather_mode,'historical_year':self._historical_year,
            'weather_contract':self._weather.contract,
            'parameter_generation_seed':PARAMETER_GENERATION_SEED,
            'parameter_identity':PARAMETER_IDENTITY,
            'information_assumptions':{
                'water_and_et_reports':'current ideal model reports in all four chapters; not a claim of perfect daily field measurements',
                'initial_state':'known research initial field capacity theta=.39, zero ponded water; not measured initial conditions',
                'weather':'policy receives shared previous-day observed rain and ET0; historical conditional mean is explicit public-display information only',
                'calendar_budget_and_terminal_evaluation':'current and public in every chapter',
                'chapter6_problem':'maximize overall outcome subject to expected positive terminal shortfall constraints for each region; no report delay',
                'observation_units':'physical units; no normalization',
                'et_policy_compression':'completed-stage Jensen product per region, active-stage actual ET per region and one shared active-stage fully supplied ET; all at received report date, complete ledgers retained for audit',
            },
        })

    def _case_info(self,info):
        # The base step has already retained a detached internal copy. These
        # fresh return values need only a new outer mapping for case metadata.
        info=dict(info)
        info.update(environment_version=CASE_ID,parameter_profile=self.parameter_profile,
                    storage_variation=self.storage_variation,rain_multiplier=self.rain_multiplier,
                    report_identity='current_model_state_reports',
                    environment_class=type(self).__name__,research_problem=self.research_problem,
                    daily_quota_m3=self.daily_quota_m3,
                    seasonal_budget_m3=ACTIVE_DAYS*self.daily_quota_m3,
                    budget_provenance=BUDGET_PROVENANCE,
                    yield_identity='Zhanghe_transferred_Jensen_not_Hongmiao_calibration',
                    maturity_identity='River2024_Sep4_reference_not_simulated',
                    evaluation_mode='terminal_projection',
                    reward_identity='area_weighted_complete_four_stage_projection_at_allocation_end',
                    outcome_stage_names=list(OUTCOME_NAMES),effective_decision_days=81,episode_physical_days=81,
                    weather_identity='shared_daily_rain_and_reference_ET0_for_all_regions',
                    late_response_limit='Jensen omits ripening stress; no postrun is simulated')
        return info

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        return obs,self._case_info(info)

    def step(self,action):
        start_day=self._day
        obs,reward,terminated,truncated,info=super().step(action)
        info=self._case_info(info)
        info.update(decision_date=(self.scenario.start_date+timedelta(days=start_day)).isoformat(),
                    evaluation_date=(self.scenario.start_date+timedelta(days=ACTIVE_DAYS)).isoformat(),
                    reference_maturity_date=(self.scenario.start_date+timedelta(days=CALENDAR_MATURITY_DAYS)).isoformat(),
                    physical_days_advanced=1,decision_steps_executed=self._day)
        return obs,reward,terminated,truncated,info

    def crop_coefficients(self, day_index=None):
        """Public dated calendar coefficients; no field truth or future weather.

        ``field_params`` holds baseline soil/model parameters. In particular,
        its baseline ``ke_wet`` is not a substitute for this dated coefficient.
        Days are physical boundaries relative to May 30; day 81 ends allocation.
        """
        index = self._day if day_index is None else day_index
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or not 0 <= index <= PHYSICAL_DAYS:
            raise ValueError('day_index must be an integer from 0 to 81')
        day = self.scenario.start_date + timedelta(days=int(index))
        profile = self._crop_profiles(day)
        return {'day_index': int(index), 'date': day.isoformat(),
                'stage_index': profile[:, 0].astype(np.int64),
                'kcb': profile[:, 1].copy(), 'ke_wet': profile[:, 2].copy(),
                'outcome_stage_index': profile[:, 3].astype(np.int64)}

    def render_context(self):
        from .hongmiao_layout import LAYOUT_ID
        result = super().render_context()
        result.update(layout_kind=LAYOUT_ID, region_ids=list(self.scenario.region_ids),
                      decision_step_index=self._day,decision_horizon_steps=ACTIVE_DAYS,
                      transition_description=('末次配置后计算模拟最终相对产量；一个物理日'
                                              if self._day == PHYSICAL_DAYS else '一次策略配置，推进一个物理日'))
        return result

    def episode_summary(self):
        result=super().episode_summary()
        result.update(case_id=CASE_ID,decision_steps_executed=min(self._day,ACTIVE_DAYS),decision_horizon_steps=self.decision_horizon_steps,physical_days_executed=self._day,parameter_profile=self.parameter_profile,daily_quota_m3=self.daily_quota_m3,seasonal_budget_m3=ACTIVE_DAYS*self.daily_quota_m3)
        return result

    def render(self):
        if self.render_mode=='ansi':
            return f'Hongmiao93 {self.parameter_profile} physical_day={self._day}/81 decision_calls={min(self._day,ACTIVE_DAYS)}/{self.decision_horizon_steps} return={self._return:.6f}'
        if self.render_mode in ('svg','rgb_array'):
            from .direct_visualization import render_observation
            context = self.render_context()
            return render_observation(self.public_observation(), self.scenario.region_ids,
                chapter=self.chapter, last_action=context['executed_action_m3'], mode=self.render_mode,
                context=context)
        return super().render()

def make_hongmiao_env(chapter=4,**kwargs):
    """Create the current 93-unit environment with exactly 81 policy steps."""
    from . import make_chapter_env
    return make_chapter_env(chapter=chapter,**kwargs)
