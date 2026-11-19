"""Fixed MSB-first RGB bit planes, with zero-filled missing suffix bits.

This is uncompressed serialization, not a learned or JPEG2000 codec. Quality
requires the source image and is used by the simulator only at termination.
"""
from __future__ import annotations

import numpy as np

IMAGE_SHAPE = (64, 64, 3)
PACKETS = 24
PAYLOAD_BYTES = 512


def encode_image(image):
    image = np.asarray(image)
    if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB[64,64,3]')
    pixels = image.reshape(-1)
    planes = ((pixels[None, :] >> np.arange(7, -1, -1)[:, None]) & 1).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder='big').reshape(PACKETS, PAYLOAD_BYTES)


def decode_prefix(received_packets):
    received_packets = np.asarray(received_packets)
    if (received_packets.ndim != 2 or received_packets.shape[1] != PAYLOAD_BYTES
            or received_packets.shape[0] > PACKETS or received_packets.dtype != np.uint8):
        raise ValueError('Expected uint8 prefix[number_of_packets<=24,512]')
    bits = np.zeros(64 * 64 * 3 * 8, dtype=np.uint8)
    received_bits = np.unpackbits(received_packets.reshape(-1), bitorder='big')
    bits[:len(received_bits)] = received_bits
    planes = bits.reshape(8, -1)
    pixels = (planes.astype(np.uint16) * (1 << np.arange(7, -1, -1))[:, None]).sum(axis=0)
    return pixels.astype(np.uint8).reshape(IMAGE_SHAPE)


def normalized_mse(source, reconstruction):
    source, reconstruction = np.asarray(source), np.asarray(reconstruction)
    if (source.shape != IMAGE_SHAPE or reconstruction.shape != IMAGE_SHAPE
            or source.dtype != np.uint8 or reconstruction.dtype != np.uint8):
        raise ValueError('MSE requires two uint8 RGB[64,64,3] images')
    return float(np.mean(((source.astype(np.float64) - reconstruction) / 255.) ** 2))
