"""Area-weighted depth projection for the explicit redundant-cap89 task.

The policy proposes relative depths x*=1+score.  With w=A/sum(A), solve
    min sum_i w_i (x_i-x*_i)^2, x>=0, sum_i w_i*x_i=1.
Then a_i=q*w_i*x_i.  This is equivalent to an area-weighted least-squares
projection in actual millimetres about h*=q/(10sum(A))*(1+score).
All physical caps equal the remaining season budget >= q, so they are
redundant; this routine deliberately rejects other cap geometries rather than
silently applying an incorrect generic bounded projector.
No target moisture, crop coefficient, weather or rule action enters here.
"""
from __future__ import annotations
import math
import numpy as np

VERSION='hongmiao89-area-weighted-depth-simplex-v1'

def weighted_depth_projection(scores, area_ha, caps, total):
    p=np.asarray(scores,dtype=np.float64)
    area=np.asarray(area_ha,dtype=np.float64)
    cap=np.asarray(caps,dtype=np.float64)
    q=float(total)
    if (p.ndim!=1 or area.shape!=p.shape or cap.shape!=p.shape or not len(p)
        or not all(np.isfinite(v).all() for v in (p,area,cap)) or
        not math.isfinite(q) or q<0 or np.any(area<=0) or np.any(cap<0)):
        raise ValueError('Finite scores, positive public areas, and nonnegative water geometry required')
    if np.any(cap<q):
        raise ValueError('This depth projector requires individually redundant caps >= total')
    if q==0:return np.zeros_like(p)
    w=area/math.fsum(area.tolist())
    proposal=1.+p
    order=np.argsort(-proposal,kind='stable')
    u=proposal[order];ws=w[order]
    cw=np.cumsum(ws);cwp=np.cumsum(ws*u)
    theta=(cwp-1.)/cw
    active=np.flatnonzero(u>theta)
    if not len(active):raise FloatingPointError('Weighted depth simplex has no free coordinate')
    shift=theta[active[-1]]
    relative_depth=np.maximum(proposal-shift,0.)
    action=q*w*relative_depth
    tolerance=2048*np.finfo(np.float64).eps*max(q,np.finfo(float).tiny)
    for _ in range(2):
        delta=q-math.fsum(action.tolist())
        if abs(delta)>tolerance:raise ArithmeticError('Depth projection mass residual exceeds roundoff')
        j=int(np.argmax(action))
        action[j]+=delta
    if (np.any(action<0) or np.any(action>cap+tolerance) or
        abs(math.fsum(action.tolist())-q)>tolerance):
        raise ArithmeticError('Depth projection violates physical geometry')
    return action
