from allocation_efficiency.frequent import FrequentPPO
from .learner import PrefixPPO
METHODS=('f1','p1','p2')
def make_method(name,spec,event_dim,seed,device):
    if name=='f1':return FrequentPPO(spec,event_dim,seed,device,replay=True)
    if name in ('p1','p2'):return PrefixPPO(spec,event_dim,seed,device,mode=name)
    raise ValueError(name)
