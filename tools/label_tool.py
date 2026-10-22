"""Hand-labelling tool for VALIDATION windows (PLAN.md 5.2). Labels never feed the submission.

    python tools/label_tool.py --name alice --start 30000 --minutes 4
    python tools/label_tool.py --name alice --start 11185 --minutes 5 --blind   # calibration on public

Plays the tactical video with the tracking drawn on it through the homography (home red, away blue,
ball yellow) and records change points. Keys:

    h / a / d     label from the current frame on: home, away, dead ball
    space         play / pause            , .   step one frame back / forward
    j l           one second back / forward      [ ]   slower / faster playback
    u             undo the last change point     s     save        q   save and quit

Saved to data/labels/<name>_<start>_<stop>.csv as change points (frame,label), and expanded to a
full frame,label file next to it so it can be scored with evaluation/run_evaluation.py --gt <file>.
Everything under data/ is gitignored: labels are made from the match video.

Calibrate first: label the public window with --blind (no ground truth shown), then
    python tools/label_tool.py --score data/labels/alice_11185_18685.csv
to see how your reading of "dead" and "owner" matches the corrected-tracking ground truth.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from possession import io  # noqa: E402

LABEL_DIR = os.path.join(io.DATA_DIR, "labels")
COLOURS = {0: (60, 60, 230), 1: (230, 120, 40)}
KEYS = {ord("h"): "H", ord("a"): "A", ord("d"): "D"}


def expand(changes, start, stop):
    """Change points -> dense frame,label over the whole video (X outside the window)."""
    labels = np.array(["X"] * io.N_FRAMES, dtype=object)
    points = sorted(changes.items())
    for i, (frame, label) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else stop
        labels[max(frame, start):min(end, stop)] = label
    return labels


def save(name, start, stop, changes):
    os.makedirs(LABEL_DIR, exist_ok=True)
    base = os.path.join(LABEL_DIR, "{}_{}_{}".format(name, start, stop))
    with open(base + "_changes.csv", "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, l) for f, l in sorted(changes.items()))
    labels = expand(changes, start, stop)
    with open(base + ".csv", "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    print("saved", base + ".csv")


def load_changes(path):
    out = {}
    if os.path.exists(path):
        with open(path) as handle:
            next(handle)
            for line in handle:
                frame, label = line.strip().split(",")
                out[int(frame)] = label
    return out


def draw(img, frame, H, players, team, ball, label, playing, speed):
    m = H[frame]
    if not np.isnan(m).any():
        for k in range(players.shape[1]):
            x, y = players[frame, k]
            if np.isnan(x):
                continue
            px, py = _project(m, x / io.PITCH[0], y / io.PITCH[1])
            cv2.circle(img, (int(px), int(py)), 9, COLOURS[int(team[k])], 2)
        bx, by = ball[frame]
        if not np.isnan(bx):
            px, py = _project(m, bx / io.PITCH[0], by / io.PITCH[1])
            cv2.circle(img, (int(px), int(py)), 6, (0, 230, 255), -1)
    text = "frame {}  label {}  {}  x{:.2g}".format(frame, label or "-", "play" if playing else "pause", speed)
    cv2.rectangle(img, (0, 0), (720, 40), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def _project(m, x, y):
    d = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / d, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / d


def label(name, start, minutes, blind):
    import load_homography
    stop = min(start + int(minutes * 60 * io.FPS), io.N_FRAMES)
    base = os.path.join(LABEL_DIR, "{}_{}_{}".format(name, start, stop))
    changes = load_changes(base + "_changes.csv")
    H, _, _ = load_homography.load()
    ball, players, team = io.load_atd()
    cap = cv2.VideoCapture(io.asset_paths()["video"])
    frame, playing, speed = start, False, 1.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    while True:
        current = [l for f, l in sorted(changes.items()) if f <= frame]
        shown = img.copy()
        draw(shown, frame, H, players, team, ball, current[-1] if current else None, playing, speed)
        cv2.imshow("label " + name + (" (blind)" if blind else ""), cv2.resize(shown, (1280, 720)))
        key = cv2.waitKey(max(1, int(40 / speed)) if playing else 0) & 0xFF
        step = 0
        if key in KEYS:
            changes[frame] = KEYS[key]
        elif key == ord(" "):
            playing = not playing
        elif key == ord("."):
            step = 1
        elif key == ord(","):
            step = -1
        elif key == ord("l"):
            step = io.FPS
        elif key == ord("j"):
            step = -io.FPS
        elif key == ord("["):
            speed = max(0.125, speed / 2)
        elif key == ord("]"):
            speed = min(8.0, speed * 2)
        elif key == ord("u") and changes:
            changes.pop(max(changes))
        elif key == ord("s"):
            save(name, start, stop, changes)
        elif key == ord("q"):
            save(name, start, stop, changes)
            break
        if playing and step == 0:
            step = 1
        if step:
            frame = int(np.clip(frame + step, start, stop - 1))
            if step != 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                playing = False
    cv2.destroyAllWindows()


def score(path):
    import possession_metrics as pm
    gt = io.load_gt_public()
    pred = pm.label_codes(pm.read_label_csv(path))
    covered = pred != io.ABSTAIN
    gt = np.where(covered, gt, io.ABSTAIN)
    res = pm.evaluate(gt, pred, bootstrap_n=0)
    print("you vs ground truth on the frames you labelled: spellF1 {:.3f}  macroF1 {:.3f}  "
          "possessions {} (truth {})".format(res["spell_f1_mean"], res["macro_f1"],
                                             res["n_pred_spells"], res["n_gt_spells"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name")
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--minutes", type=float, default=4.0)
    parser.add_argument("--blind", action="store_true", help="nothing from the ground truth is shown (it never is; flag kept for the record)")
    parser.add_argument("--score", help="score a saved labels file against the public ground truth")
    args = parser.parse_args()
    if args.score:
        score(args.score)
    else:
        label(args.name, args.start, args.minutes, args.blind)


if __name__ == "__main__":
    main()
