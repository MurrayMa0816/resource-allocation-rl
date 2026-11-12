"""Construct a real 89-object scenario before physical state allocation."""
import copy
import json
import math
from datetime import date, timedelta
from numbers import Integral, Real
import numpy as np
from irrigation_ch6._core.hongmiao import (
    _inputs, ACTIVE_DAYS, DEFAULT_DAILY_QUOTA_M3, DEFAULT_STORAGE_VARIATION,
    FIXED_FIELD_PARAMETERS, FROZEN_PARAMETER_TABLE, INITIAL_THETA,
    PARAMETER_GENERATION_SEED, PARAMETER_IDENTITY, STORAGE_GENERATION_SEED,
    REFERENCE_ROOT_AVAILABLE_WATER_MM)
from irrigation_ch6._core.direct_scenarios import DirectScenario
from irrigation_ch6._core.field import FieldParams
from irrigation_ch6._core.weather import WeatherDay, PARAM_NAMES
from .constants import (CASE_ID, EXCLUDED_IDS, DEFAULT_BUDGET_M3, BUDGET_FORMULA,
                        RETAINED_AREA_MU, PARENT_AREA_MU, RETAINED_IDS)

def load_hongmiao89_scenario(*,profile='heterogeneous',historical_year=2022,
                          daily_quota_m3=DEFAULT_BUDGET_M3 / ACTIVE_DAYS,
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
    parent_ids = tuple(r['region_id'] for r in rows)
    rows = [r for r in rows if r['region_id'] not in EXCLUDED_IDS]
    if len(rows) != 89 or set(EXCLUDED_IDS) - set(parent_ids):
        raise ValueError('Reduced case must remove exactly the four registered IDs')
    if tuple(r['region_id'] for r in rows) != RETAINED_IDS:
        raise ValueError('Source region order differs from the registered reduced subset')
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
    if not np.isclose(area.sum(),RETAINED_AREA_MU/15,rtol=0,atol=1e-9):
        raise ValueError('Hongmiao published total rice area is inconsistent')
    # No rescaling: dropping rows changes the subset's weighted mean naturally.
    # Every retained field preserves the parent's exact frozen parameter rule.
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
        'budget_provenance':'explicit reduced-case budget; default '+BUDGET_FORMULA,
        'budget_contract':{'parent_input_contract':budget,'effective_total_budget_m3':ACTIVE_DAYS*daily_quota_m3,'default_formula':BUDGET_FORMULA,'default_total_budget_m3':DEFAULT_BUDGET_M3},
        'daily_supply_limit_m3':ACTIVE_DAYS*daily_quota_m3,
        'daily_supply_limit_identity':'seasonal budget as mathematical upper bound; ideal availability, not engineering capacity',
        'regional_delivery_capacity':'not imposed; no calibrated regional engineering capacity',
        'retained_rice_area_ha':RETAINED_AREA_MU/15, 'parent_published_rice_area_ha':PARENT_AREA_MU/15,
        'excluded_region_ids':list(EXCLUDED_IDS), 'parent_region_ids':list(parent_ids),
        'region_selection':'four fixed IDs removed before FieldParams and DirectScenario construction',
        'case_status':'reduced research subset; not the complete published 93-unit case',
        'geography':'Hongmiao irrigation service units, Zhanghe district, Hubei; polygon geometries unavailable',
        'outcome':'same-district transferred four-stage Jensen ET response; not measured harvest',
        'evaluation':'terminal model projection at allocation end; no postrun or claim of biological maturity',
        'omitted_ripening':'the four-stage response does not model later ripening stress',
        'sources':copy.deepcopy(manifest['sources']),'manifest':manifest,
    }
    scenario=DirectScenario(region_ids=tuple(r['region_id'] for r in rows),field_params=tuple(params),initial_storage_mm=np.array(initial),heading_dates=tuple(date(historical_year,7,31) for _ in rows),quota_plan=quota,evaluation_mode="terminal_projection",start_date=date(historical_year,5,30),weather_records=records,donor_pools=pools,forecast_table=forecasts,weather_contract=prior,maturity_offset_days=35,field_substeps_per_day=96,metadata=metadata)
    return scenario,model
