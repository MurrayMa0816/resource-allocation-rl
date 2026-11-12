"""Avoid rehashing an unchanged 331 MB image file on every native replay.

Execution-only: validate once, check exact filesystem identity on every lookup,
and rehash at the end. No state, transition, reward, model or RNG changes.
"""
from pathlib import Path
import hashlib
import time

def fingerprint(path):
    s=Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def full_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

class FrozenDigest:
    def __init__(self,path,original):
        self.path=Path(path).resolve()
        self.original=original
        self.identity=fingerprint(self.path)
        self.expected=original(self.path)
        if fingerprint(self.path)!=self.identity:
            raise RuntimeError('Frozen image input changed during initial validation')
        self.hits=0

    def __call__(self,path):
        resolved=Path(path).resolve()
        if resolved!=self.path:return self.original(path)
        if fingerprint(resolved)!=self.identity:
            raise RuntimeError('Frozen image input filesystem identity changed')
        self.hits+=1
        return self.expected

    def verify_final(self):
        assert fingerprint(self.path)==self.identity
        assert self.original(self.path)==self.expected
        return dict(path=str(self.path),sha256=self.expected,cache_hits=self.hits,
                    final_full_rehash_passed=True,filesystem_identity=list(self.identity))

def install(image_path=None):
    import allocation23.task as task
    path=Path(image_path) if image_path else task.DATA_ROOT/'images.npy'
    if isinstance(task.digest,FrozenDigest):return task.digest
    cached=FrozenDigest(path,task.digest)
    task.digest=cached
    return cached
