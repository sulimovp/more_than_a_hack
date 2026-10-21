"""Does the tactical video's audio show referee whistles at the public window's stoppages?

    python analysis/whistle_probe.py

Narrow-band energy in 2.5-4.5 kHz (pea whistles sit around 3-4 kHz) relative to the broadband
level, one value per video frame. Prints the peaks and the ground-truth transitions beside them.
"""

import os
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import asset_paths  # noqa: E402

FPS = 25
FIRST, LAST = 11185 - 250, 18685 + 250


def audio_window(path, t0, t1):
    container = av.open(path)
    stream = container.streams.audio[0]
    rate = stream.sample_rate
    container.seek(int(t0 / stream.time_base), stream=stream)
    chunks, start = [], None
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        if t > t1:
            break
        data = frame.to_ndarray()
        data = data.mean(axis=0) if data.ndim == 2 else data
        if start is None:
            start = t
        chunks.append(data.astype(np.float32))
    return np.concatenate(chunks), rate, start


def band_ratio_per_frame(signal, rate, start, first_frame, n_frames):
    hop = rate // FPS
    window = np.hanning(2048)
    freqs = np.fft.rfftfreq(2048, 1.0 / rate)
    band = (freqs >= 2500) & (freqs <= 4500)
    rest = (freqs >= 300) & (freqs <= 8000) & ~band
    out = np.full(n_frames, np.nan)
    for i in range(n_frames):
        centre = int(((first_frame + i) / FPS - start) * rate)
        seg = signal[centre - 1024:centre + 1024]
        if len(seg) < 2048:
            continue
        spec = np.abs(np.fft.rfft(seg * window)) ** 2
        # peakiness inside the band: a whistle is a narrow tone, crowd noise is broad
        out[i] = 10 * np.log10(spec[band].max() / (spec[rest].mean() + 1e-12) + 1e-12)
    return out


def main():
    path = asset_paths()["video"]
    signal, rate, start = audio_window(path, FIRST / FPS, LAST / FPS)
    score = band_ratio_per_frame(signal, rate, start, FIRST, LAST - FIRST)
    gt = pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    trans = [t for t in pm.transitions(gt, pm.valid_runs(gt))]
    thr = np.nanpercentile(score, 99)
    print("score p50 {:.1f} dB, p99 {:.1f} dB".format(np.nanpercentile(score, 50), thr))
    peaks = np.flatnonzero(score > thr) + FIRST
    events = []
    for f in peaks:
        if not events or f - events[-1][1] > 12:
            events.append([f, f])
        else:
            events[-1][1] = f
    print("loud narrow-band events (frame range):")
    for a, b in events:
        near = [(t, "HADX"[x] + ">" + "HADX"[y]) for t, x, y in trans if abs(t - a) < 150]
        print("  {}-{}  {}".format(a, b, near))
    print("dead-ball transitions in the public window:")
    for t, x, y in trans:
        if 2 in (x, y):
            print("  {} {}>{}  max score within -2s..+1s: {:.1f}".format(
                t, "HADX"[x], "HADX"[y], np.nanmax(score[t - 50 - FIRST:t + 25 - FIRST])))


if __name__ == "__main__":
    main()
