"""Source identities include the new adapter and immutable dependency hashes."""
from pathlib import Path
import hashlib
from irrigation_ch6._util import digest, jsonable, real


def package_files():
    root = Path(__file__).parent
    result = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.iterdir()) if p.suffix == '.py' or p.name == 'dependency_manifest.json'}
    entry = root.parent / 'irrigation89.py'
    result['../irrigation89.py'] = hashlib.sha256(entry.read_bytes()).hexdigest()
    return result
