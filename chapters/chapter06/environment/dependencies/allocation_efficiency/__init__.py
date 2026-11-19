"""Public-data sample reuse on the frozen chapter23 A task."""
import sys
for _p in ['/root/gamewar/chapter4_credit_assignment','/root/gamewar']:
    if _p not in sys.path: sys.path.insert(0,_p)
import allocation4  # pins allocation23 to the released standalone package
