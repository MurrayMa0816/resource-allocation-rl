from .baseline import NativePPO
METHODS=('b1','f1','p2','s2')

def make_method(name,spec,event_dim,seed,device):
    if name=='b1':
        return NativePPO(spec,event_dim,seed,device,extra=True)
    if name=='f1':
        from .frequent import FrequentPPO
        return FrequentPPO(spec,event_dim,seed,device,replay=True)
    if name in ('p2','s2'):
        from .calibrated import CalibratedSnapshotPG
        return CalibratedSnapshotPG(spec,event_dim,seed,device,variance_reduced=name=='s2')
    raise ValueError(name)
