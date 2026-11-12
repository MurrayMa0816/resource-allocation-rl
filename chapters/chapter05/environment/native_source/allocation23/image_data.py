"""Read-only image episode generation, owned by the experiment/task generator.

Source pixels, IDs, labels and quality caches are never policy observations.
This module reads the bundled prepared EuroSAT RGB dataset without modifying it.  Splits are the project's fixed, class-stratified
research split, not a claim of geographic generalization.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
import os

import numpy as np


DATA_ROOT = Path(os.environ.get('ALLOCATION23_DATA_ROOT', str(Path(__file__).resolve().parent / 'data' / 'eurosat_rgb')))
EPISODE_SEED = 2026102906
SPLIT_CODES = {'train': 0, 'validation': 1, 'test': 2}
IMAGE_SHAPE = (64, 64, 3)


def _nonnegative_integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer.')
    value = int(value)
    if value < (1 if positive else 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}.')
    return value


def _images(root):
    path = Path(root) / 'images.npy'
    images = np.load(path, mmap_mode='r', allow_pickle=False)
    if (images.dtype != np.uint8 or images.ndim != 4
            or images.shape[1:] != IMAGE_SHAPE or not len(images)):
        raise ValueError('images.npy must contain nonempty uint8 RGB images of shape (N, 64, 64, 3).')
    return images


def _split_pools(manifest, count):
    splits = manifest.get('splits')
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_CODES):
        raise ValueError('Manifest must define exactly train, validation and test splits.')
    pools = {}
    seen = set()
    for name in SPLIT_CODES:
        entries = splits[name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'{name} split must be a nonempty list of image IDs.')
        ids = [_nonnegative_integer(item, f'{name} image ID') for item in entries]
        unique = set(ids)
        if max(ids) >= count:
            raise ValueError(f'{name} split contains an out-of-range image ID.')
        if len(unique) != len(ids):
            raise ValueError(f'{name} split contains duplicate image IDs.')
        if seen.intersection(unique):
            raise ValueError('Image IDs must be disjoint across train, validation and test.')
        seen.update(unique)
        pools[name] = np.asarray(ids, dtype=np.int64)
        pools[name].flags.writeable = False
    declared_counts = manifest.get('counts')
    if declared_counts is not None and declared_counts != {k: len(v) for k, v in pools.items()}:
        raise ValueError('Manifest counts disagree with the split lists.')
    if manifest.get('archive_images', count) != count:
        raise ValueError('Manifest archive_images disagrees with images.npy.')
    return pools


class ImageEpisodes:
    """Deterministic episode plans, separate from both policy and simulator.

    Images are distinct within an episode and may recur in later episodes.
    Two independent SeedSequence children select images and channel seed, so
    changing the number of receivers does not change the channel seed.  The
    default plan preserves the existing experiment's IDs and channel seeds.

    ``plan`` is experiment metadata.  ``reset`` forwards only the environment's
    own observation/info and never augments them with IDs or source pixels.
    The environment must copy its source pixels when resetting, keeping a fixed
    image for each receiver for the entire episode.
    """

    def __init__(self, split, groups=100, root=DATA_ROOT, *, master_seed=EPISODE_SEED):
        if split not in SPLIT_CODES:
            raise ValueError(f'Unknown image split {split!r}; choose {tuple(SPLIT_CODES)}.')
        self.groups = _nonnegative_integer(groups, 'groups', positive=True)
        self.master_seed = _nonnegative_integer(master_seed, 'master_seed')
        self.root = Path(root)
        self.split = split
        self.manifest = json.loads((self.root / 'split_manifest.json').read_text())
        self.images = _images(self.root)
        self.pool = _split_pools(self.manifest, len(self.images))[split]
        if self.groups > len(self.pool):
            raise ValueError('Not enough unique source images for this episode.')
        # No prefix_mse.npy is loaded here: that file is privileged evaluation
        # data, unnecessary for ordinary image sampling and environment reset.

    def plan(self, episode):
        episode = _nonnegative_integer(episode, 'episode')
        image_ss, channel_ss = np.random.SeedSequence(
            [self.master_seed, SPLIT_CODES[self.split], episode]).spawn(2)
        ids = np.random.default_rng(image_ss).choice(self.pool, self.groups, replace=False)
        channel_seed = int(channel_ss.generate_state(1)[0])
        return ids, channel_seed

    def sample(self, rng):
        """Supply pixels to an environment-owned, independent reset RNG.

        ``CommunicationImageEnv(image_source=episodes.sample)`` can use this
        callable to support ordinary Gymnasium ``reset(seed=...)``.  For exact
        historical paired paths use ``plan``/``reset`` instead: the environment's
        image-source stream and the historical episode-plan stream are distinct.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError('Image sampling requires a numpy.random.Generator.')
        ids = rng.choice(self.pool, self.groups, replace=False)
        return self.images[ids]

    def reset(self, env, episode):
        ids, channel_seed = self.plan(episode)
        # Advanced indexing creates a per-episode array; no writable reference
        # to the shared dataset is passed to the environment.
        pixels = self.images[ids]
        return env.reset(seed=channel_seed, options={'images': pixels})

