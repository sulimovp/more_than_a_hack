"""Full-match facts the plan depends on: ball gaps, player speed, stuck ball, set pieces.

    python analysis/full_match_stats.py

Reads the cache from analysis/cache_atd.py and the public ground truth. Prints numbers only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
import load_smart_tagging  # noqa: E402
from cache_atd import load  # noqa: E402
from games import half_video_blocks  # noqa: E402

FPS = 25
PITCH = (105.3, 68.0)
PUBLIC = (11185, 18685)


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def collective_speed(players):
    """Median over visible players of displacement over 1 s, in m/s, centred on each frame."""
    half = FPS // 2
    shifted_fwd = np.roll(players, -half, axis=0)
    shifted_back = np.roll(players, half, axis=0)
    step = np.linalg.norm(shifted_fwd - shifted_back, axis=2) * FPS / (2 * half)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        return np.nanmedian(step, axis=1)


def auc(pos, neg):
    """P(score of a random positive > random negative), ties half."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ball, players, team = load()
    speed = collective_speed(players)
    visible = ~np.isnan(ball[:, 0])

    print("== ball gaps per graded half ==")
    bins = [0, 1, 3, 5, 10, 30, 1e9]
    for half, first, last in half_video_blocks("BAR-ATH"):
        sl = slice(first, last + 1)
        gaps = [(a, b) for a, b in runs_of(~visible[sl])]
        lengths = np.array([b - a for a, b in gaps]) / FPS
        frames = np.array([b - a for a, b in gaps])
        print("half {}: visible {:.1%}, {} gaps".format(half, visible[sl].mean(), len(gaps)))
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (lengths >= lo) & (lengths < hi)
            print("   gaps {:>4}-{:<5}s: n={:4d}  share of half frames {:.1%}".format(
                lo, hi if hi < 1e9 else "inf", sel.sum(), frames[sel].sum() / (last - first + 1)))

    print("\n== collective speed (median player m/s) ==")
    gt = pm.label_codes(pm.read_label_csv(
        os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    win = slice(*PUBLIC)
    g, s, v = gt[win], speed[win], visible[win]
    dead, live = s[g == 2], s[g < 2]
    print("public: dead median {:.2f}, live median {:.2f}, AUC(live faster) {:.3f}".format(
        np.nanmedian(dead), np.nanmedian(live), auc(live, dead)))
    print("public, ball missing only: dead n={} live n={}, AUC {:.3f}".format(
        np.sum((g == 2) & ~v), np.sum((g < 2) & ~v), auc(s[(g < 2) & ~v], s[(g == 2) & ~v])))
    for half, first, last in half_video_blocks("BAR-ATH"):
        sl = slice(first, last + 1)
        sv, ss = visible[sl], speed[sl]
        print("half {}: speed when ball visible p25/50/75 {:.2f}/{:.2f}/{:.2f}; "
              "in gaps >5 s {:.2f}/{:.2f}/{:.2f}; gap frames below 1.4 m/s {:.1%}".format(
                  half, *np.nanpercentile(ss[sv], [25, 50, 75]),
                  *np.nanpercentile(ss[~sv], [25, 50, 75]), np.nanmean(ss[~sv] < 1.4)))

    print("\n== out of pitch and stuck ball (graded, ball visible) ==")
    graded = np.zeros(len(ball), bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    bx, by = ball[:, 0], ball[:, 1]
    out = visible & ((bx < 0) | (bx > PITCH[0]) | (by < 0) | (by > PITCH[1]))
    print("out of pitch: {:.1%} of visible graded frames".format(out[graded].sum() / (visible & graded).sum()))
    moved = np.linalg.norm(ball[FPS:] - ball[:-FPS], axis=1)
    moved = np.concatenate([moved, np.full(FPS, np.nan)])
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    import warnings
    warnings.simplefilter("ignore")
    nearest = np.nanmin(d, axis=1)
    stuck = visible & graded & (moved < 0.3) & (nearest > 5)
    stuck_runs = [(a, b) for a, b in runs_of(stuck) if b - a >= 2 * FPS]
    print("stuck ball (still >=2 s, nobody within 5 m): {} runs, {} frames".format(
        len(stuck_runs), sum(b - a for a, b in stuck_runs)))
    pub_stuck = [(a, b) for a, b in stuck_runs if PUBLIC[0] <= a < PUBLIC[1]]
    for a, b in pub_stuck:
        print("   public stuck run {}-{}: gt {}".format(a, b, np.unique(gt[a:b])))

    print("\n== set pieces (Smart Tagging) ==")
    events = load_smart_tagging.load(fps=FPS)
    from collections import Counter
    sp = [e for e in events if e["code"] in ("SET PIECES", "GOALS")]
    print(Counter((e["code"], e["tags"].get("Type", ""), e["tags"].get("Team", "")) for e in sp))
    vis_before = [visible[max(e["start_frame"] - 250, 0):e["start_frame"]].mean() for e in sp]
    print("ball visible in the 10 s before a set-piece tag: median {:.0%}".format(np.median(vis_before)))
    n_public_dead = len([1 for a, b, k in pm.segments(gt, pm.valid_runs(gt)) if k == 2])
    print("public window dead spells: {} in 5 min -> ~{:.0f} per 106 min if the rate held".format(
        n_public_dead, n_public_dead * 106 / 5))


if __name__ == "__main__":
    main()
