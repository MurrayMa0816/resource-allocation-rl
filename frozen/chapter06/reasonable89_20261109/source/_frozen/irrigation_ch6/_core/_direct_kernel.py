"""Optional compiled fp64 daily kernel; no cache files in the source tree.

Input validation and all scientific checks remain in direct_execution. Both
backends retain every numerical substep and return identical physical fields.
Numba is optional; importing the environment needs only NumPy.
"""
import math
import numpy as np


def numpy_substeps(B, rain_step, irrigation_step, maximum, saturation, wilting, denominator, tpot, epot, capacity, drain_rate, drain_power, evap_power, steps):
    B = B.copy()
    daily = {key: np.zeros(len(B)) for key in ('Ta', 'Ea', 'D', 'Q')}
    for _ in range(steps):
        # Compensated sum reduces deviation from scalar math.fsum inflows.
        first = B + rain_step
        z = first - B
        error = (B - (first - z)) + (rain_step - z)
        second = first + irrigation_step
        z = second - first
        error += (first - (second - z)) + (irrigation_step - z)
        before = second + error
        if not np.isfinite(before).all():
            raise ValueError("combined water input must be finite")
        Q = np.maximum(before - maximum, 0.)
        B = np.minimum(before, maximum)
        W = np.minimum(B, saturation)
        ks = np.clip((W - wilting) / denominator, 0., 1.)
        wetness = np.where(B > saturation, 1., (W / saturation) ** evap_power)
        Ta = np.minimum(ks * tpot, np.maximum(B - wilting, 0.))
        B -= Ta
        Ea = np.minimum(epot * wetness, B)
        B -= Ea
        fraction = np.clip((np.minimum(B, saturation) - capacity) / (saturation - capacity), 0., 1.) ** drain_power
        D = np.minimum(np.maximum(B - capacity, 0.), drain_rate * fraction)
        B -= D
        daily["Ta"] += Ta; daily["Ea"] += Ea; daily["D"] += D; daily["Q"] += Q
    return B, daily['Ta'], daily['Ea'], daily['D'], daily['Q']


def _scalar_substeps(B, rain_step, irrigation_step, maximum, saturation, wilting, denominator, tpot, epot, capacity, drain_rate, drain_power, evap_power, steps):
    B = B.copy()
    ta = np.zeros(len(B), dtype=np.float64)
    ea = np.zeros(len(B), dtype=np.float64)
    drainage = np.zeros(len(B), dtype=np.float64)
    overflow = np.zeros(len(B), dtype=np.float64)
    for i in range(len(B)):
        water = B[i]
        for _ in range(steps):
            # Same compensated inflow arithmetic as the NumPy reference. Never
            # enable fastmath: reassociation would invalidate this compensation.
            first = water + rain_step[i]
            z = first - water
            error = (water - (first-z)) + (rain_step[i]-z)
            second = first + irrigation_step[i]
            z = second-first
            error += (first-(second-z)) + (irrigation_step[i]-z)
            before = second + error
            if not math.isfinite(before):
                raise ValueError('combined water input must be finite')
            Q = max(before-maximum[i], 0.)
            water = min(before, maximum[i])
            W = min(water, saturation[i])
            ks = min(max((W-wilting[i])/denominator[i], 0.), 1.)
            wetness = 1. if water > saturation[i] else (W/saturation[i])**evap_power[i]
            Ta = min(ks*tpot[i], max(water-wilting[i], 0.))
            water -= Ta
            Ea = min(epot[i]*wetness, water)
            water -= Ea
            fraction = min(max((min(water,saturation[i])-capacity[i])/(saturation[i]-capacity[i]), 0.), 1.)**drain_power[i]
            D = min(max(water-capacity[i],0.), drain_rate[i]*fraction)
            water -= D
            ta[i] += Ta; ea[i] += Ea; drainage[i] += D; overflow[i] += Q
        B[i] = water
    return B, ta, ea, drainage, overflow


try:
    from numba import njit
except ImportError:
    compiled_substeps = None
else:
    compiled_substeps = njit(cache=False, fastmath=False, nogil=True, error_model='numpy')(_scalar_substeps)

BACKEND = 'numba' if compiled_substeps is not None else 'numpy'
field_substeps = compiled_substeps if compiled_substeps is not None else numpy_substeps
