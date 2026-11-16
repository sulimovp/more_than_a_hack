"""Offline frame-level labelling of random one-minute windows (validation only; labels never feed the file).

    python tools/window_offline.py --name pavel                  # data/offline_review/<name>_windows.zip
    python tools/window_offline.py --name pavel --merge pasted.txt
    python tools/window_offline.py --name pavel --score          # the submitted file against the windows

Transition review measures how many predicted changes are real; it cannot see the changes the file
misses. Labelling whole windows can. WINDOWS one-minute windows are drawn with a fixed seed from the
graded blocks outside the public window, each rendered as a video (broadcast | tactical with
tracking). The labeller marks change points (H, A, D) on a page that works offline; the pasted
change points are saved as data/labels/windows_<name>.csv and scored with the official metric on
the window frames only.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from possession import io  # noqa: E402
from review_offline import OUT_DIR, SIZE, encode  # noqa: E402
from review_web import PUBLIC_WINDOW, Clips  # noqa: E402

WINDOWS = 8
LENGTH = 60 * io.FPS
SEED = 1117
LABEL_DIR = os.path.join(io.DATA_DIR, "labels")
CSV = os.path.join(ROOT, "evaluation", "results", "possession_pipeline.csv")


def windows():
    """[(lo, hi)] fixed-seed one-minute windows inside the graded blocks, outside the public window,
    not overlapping each other."""
    rng = np.random.default_rng(SEED)
    blocks = [(first, stop) for _, first, stop in io.halves()]
    out = []
    while len(out) < WINDOWS:
        first, stop = blocks[int(rng.integers(len(blocks)))]
        lo = int(rng.integers(first, stop - LENGTH))
        hi = lo + LENGTH
        if hi > PUBLIC_WINDOW[0] and lo < PUBLIC_WINDOW[1]:
            continue
        if any(lo < b and a < hi for a, b in out):
            continue
        out.append((lo, hi))
    return sorted(out)


def build(name):
    clips = Clips([])
    folder = os.path.join(OUT_DIR, "{}_windows".format(name))
    os.makedirs(os.path.join(folder, "clips"), exist_ok=True)
    items = []
    for k, (lo, hi) in enumerate(windows()):
        clip = "clips/window_{}.mp4".format(lo)
        if not os.path.exists(os.path.join(folder, clip)):
            encode(os.path.join(folder, clip), clips.composite_range(lo, hi, SIZE))
        items.append({"lo": lo, "hi": hi, "clip": clip})
        print("window {}/{} at {}".format(k + 1, WINDOWS, lo), flush=True)
    page = open(os.path.join(HERE, "window_offline.html"), encoding="utf-8").read()
    page = page.replace("/*MANIFEST*/null", json.dumps({"name": name, "fps": io.FPS, "items": items}))
    with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(page)
    shutil.make_archive(folder, "zip", folder)
    print("wrote", folder + ".zip")


def windows_path(name):
    return os.path.join(LABEL_DIR, "windows_{}.csv".format(name))


def merge(name, path):
    """Pasted lines window_lo,frame,label are kept; a later paste for a window replaces that window."""
    rows = {}
    if os.path.exists(windows_path(name)):
        with open(windows_path(name)) as handle:
            next(handle)
            for line in handle:
                lo, frame, label = line.strip().split(",")
                rows.setdefault(int(lo), []).append((int(frame), label))
    new = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) == 3 and parts[0].isdigit() and parts[2] in ("H", "A", "D"):
                new.setdefault(int(parts[0]), []).append((int(parts[1]), parts[2]))
    rows.update(new)
    os.makedirs(LABEL_DIR, exist_ok=True)
    with open(windows_path(name), "w", newline="\n") as handle:
        handle.write("window_lo,frame,label\n")
        for lo in sorted(rows):
            handle.writelines("{},{},{}\n".format(lo, f, l) for f, l in sorted(rows[lo]))
    print("windows labelled:", len(rows), "->", windows_path(name))


def dense(name):
    """Codes over the whole video: the labelled windows, ABSTAIN elsewhere (and before a window's first mark)."""
    codes = np.full(io.N_FRAMES, io.ABSTAIN, dtype=np.int8)
    code = {"H": io.HOME, "A": io.AWAY, "D": io.DEAD}
    span = dict(windows())
    rows = {}
    with open(windows_path(name)) as handle:
        next(handle)
        for line in handle:
            lo, frame, label = line.strip().split(",")
            rows.setdefault(int(lo), []).append((int(frame), label))
    for lo, marks in rows.items():
        hi = span.get(lo, lo + LENGTH)
        marks = sorted(marks)
        for i, (f, l) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else hi
            codes[max(f, lo):min(end, hi)] = code[l]
    return codes


def score(name, csv=CSV):
    import possession_metrics as pm
    truth = dense(name)
    pred = pm.label_codes(pm.read_label_csv(csv))
    res = pm.evaluate(truth, pred, bootstrap_n=200)
    print("{} labelled frames".format(int((truth != io.ABSTAIN).sum())))
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict, np.ndarray))})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--merge")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--csv", default=CSV)
    args = parser.parse_args()
    if args.merge:
        merge(args.name, args.merge)
    elif args.score:
        score(args.name, args.csv)
    else:
        build(args.name)


if __name__ == "__main__":
    main()
