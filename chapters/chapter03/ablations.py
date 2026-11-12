"""The three published Chapter 3 ablations; legacy arms are explicit."""
from pathlib import Path
import json


def registry():
    return json.loads((Path(__file__).resolve().parent/'configs/ablations.json').read_text())


def get(arm):
    for entry in registry()['arms']:
        if entry['id'] == arm:
            return dict(entry)
    raise ValueError('Unknown Chapter 3 ablation: '+str(arm))
