"""S2a: referee-whistle score per video frame from the tactical video's audio track.

A pea whistle is a narrow tone around 3-4 kHz; crowd noise is broadband. The score is the peak
power inside 2.5-4.5 kHz over the mean power of the rest of 0.3-8 kHz, in dB, from a 2048-sample
Hann window centred on each video frame. Crowd whistling also scores, so this is evidence for the
decoder, never a rule. Cached in data/cache/.
"""

import os

import av
import numpy as np

from .io import FPS, N_FRAMES, asset_paths, cache_path

WINDOW = 2048
BAND = (2500.0, 4500.0)
REST = (300.0, 8000.0)


def decode_mono(path):
    container = av.open(path)
    stream = container.streams.audio[0]
    rate = stream.sample_rate
    chunks, start = [], None
    for frame in container.decode(stream):
        data = frame.to_ndarray()
        if data.ndim == 2:
            data = data.mean(axis=0)
        if start is None:
            start = float(frame.pts * stream.time_base)
        chunks.append(data.astype(np.float32))
    return np.concatenate(chunks), rate, start


def whistle_score():
    path = cache_path("whistle_score.npy")
    if os.path.exists(path):
        return np.load(path)
    signal, rate, start = decode_mono(asset_paths()["video"])
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / rate)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    rest = (freqs >= REST[0]) & (freqs <= REST[1]) & ~band
    hann = np.hanning(WINDOW).astype(np.float32)
    score = np.full(N_FRAMES, np.nan, dtype=np.float32)
    centres = ((np.arange(N_FRAMES) / FPS - start) * rate).astype(np.int64)
    ok = np.flatnonzero((centres >= WINDOW // 2) & (centres + WINDOW // 2 <= len(signal)))
    for chunk in np.array_split(ok, max(1, len(ok) // 4096)):
        idx = centres[chunk][:, None] + np.arange(-WINDOW // 2, WINDOW // 2)[None, :]
        spec = np.abs(np.fft.rfft(signal[idx] * hann, axis=1)) ** 2
        ratio = spec[:, band].max(axis=1) / (spec[:, rest].mean(axis=1) + 1e-12)
        score[chunk] = 10 * np.log10(ratio + 1e-12)
    np.save(path, score)
    return score
