"""Render video frames with the tracking drawn on them, to look at one moment without a GUI.

    python analysis/render_frames.py 15762 15770 15778 15786 15794 15802 --out seq.png
    python analysis/render_frames.py 17313 --width 600 --height 360

Writes one PNG with the frames side by side (three per row), each cropped around the ATD ball
and enlarged twice. Home slots are red crosses, away slots blue, the ATD ball a yellow cross, the
slot index next to each player; the public ground-truth label is in the corner. The output is
derived from the match video, so keep it out of git (write it under data/ or a temp folder).

This is what settled the 2 Nov review (ROADMAP.md section 1): an agent can read the frames
with its image tool, and a person can read them in any viewer. Needs opencv-python.
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

import load_homography  # noqa: E402

HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = (60, 60, 230), (230, 120, 40), (0, 230, 230)


def render(frames, width=420, height=260, per_row=3):
    ball, players, team = io.load_atd()
    H, _, _ = load_homography.load()
    gt = io.load_gt_public()
    cap = cv2.VideoCapture(io.asset_paths()["video"])
    tiles = []
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            raise IOError("could not read frame {}".format(frame))
        m = H[frame]

        def project(x, y):
            p = m @ np.array([x / io.PITCH[0], y / io.PITCH[1], 1.0])
            return int(p[0] / p[2]), int(p[1] / p[2])

        for k in range(players.shape[1]):
            x, y = players[frame, k]
            if np.isnan(x):
                continue
            u, v = project(x, y)
            cv2.drawMarker(img, (u, v), HOME_COLOUR if team[k] == io.HOME else AWAY_COLOUR,
                           cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(img, str(k), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bx, by = ball[frame]
        if np.isnan(bx):
            u, v = img.shape[1] // 2, img.shape[0] // 2
        else:
            u, v = project(bx, by)
            cv2.drawMarker(img, (u, v), BALL_COLOUR, cv2.MARKER_CROSS, 16, 1)
        x0 = min(max(u - width // 2, 0), img.shape[1] - width)
        y0 = min(max(v - height // 2, 0), img.shape[0] - height)
        crop = cv2.resize(img[y0:y0 + height, x0:x0 + width], (width * 2, height * 2),
                          interpolation=cv2.INTER_CUBIC)
        text = "{} gt {}".format(frame, "HADX"[gt[frame]])
        cv2.putText(crop, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(crop, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        tiles.append(crop)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=int, nargs="+")
    parser.add_argument("--out", default=os.path.join(io.DATA_DIR, "frames.png"))
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=260)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, render(args.frames, args.width, args.height))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
