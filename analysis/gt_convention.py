"""Where does the public ground truth put a team switch, relative to the automatic ball track?

    python analysis/gt_convention.py

For each H<->A switch in the public window: when the new team first comes within 1.5 m of the
ATD ball, how far both teams are at the switch frame, and the ball speed. Then the opposite:
stretches where the non-possessing team is within 1 m for 0.4 s or more and the ground truth
does not switch. On 2 Nov this showed 5 of 12 switches with the new team never within 1 m in the
following 2 s, and 6 ignored contacts (ROADMAP.md section 1). Render any of them with
analysis/render_frames.py before drawing a conclusion: three turned out to be the ATD ball being
wrong (a header in the air, a false detection), not the ground truth.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession import control, io, pipeline  # noqa: E402
from possession.ball import runs  # noqa: E402

import possession_metrics as pm  # noqa: E402

PUBLIC = (11185, 18685)


def main():
    f = pipeline.features()
    b, players, team = f["ball"], f["players"], f["team"]
    gt = io.load_gt_public()
    dist, _ = control._closest(b, players, team)
    speed = np.nan_to_num(b["speed"], nan=0.0)
    trans = [(fr, a, c) for fr, a, c in pm.transitions(gt, pm.valid_runs(gt)) if PUBLIC[0] < fr < PUBLIC[1]]

    print("GT team switches: when the new team first comes within 1.5 m of the ATD ball")
    for fr, a, c in trans:
        if a > 1 or c > 1:
            continue
        win = np.arange(fr - 50, fr + 50)
        near = win[dist[win, c] <= 1.5]
        first = "{:+d} frames".format(int(near[0] - fr)) if len(near) else "never in +/-2 s"
        before = np.arange(fr - 75, fr)
        last_old = before[dist[before, a] <= 1.5]
        print("  {} {}>{}: new team within 1.5 m {}; dist new/old at the switch {:.2f}/{:.2f} m; "
              "ball {:.1f} m/s; old team last within 1.5 m {} frames before".format(
                  fr, "HA"[a], "HA"[c], first, dist[fr, c], dist[fr, a], speed[fr],
                  int(fr - last_old[-1]) if len(last_old) else ">75"))

    print("\nContacts the GT ignores: the other team within 1 m for >= 0.4 s, the possessing team > 1.5 m")
    sl = slice(*PUBLIC)
    g = gt[sl]
    for side in (0, 1):
        other = 1 - side
        mask = (g == side) & (dist[sl, other] <= 1.0) & (dist[sl, side] > 1.5) & b["visible"][sl]
        for a, z in runs(mask):
            if z - a >= 10:
                print("  GT {} but {} within 1 m for {:.2f} s at {}-{}, ball {:.1f}-{:.1f} m/s".format(
                    "HA"[side], "HA"[other], (z - a) / 25.0, a + PUBLIC[0], z + PUBLIC[0],
                    speed[sl][a:z].min(), speed[sl][a:z].max()))


if __name__ == "__main__":
    main()
