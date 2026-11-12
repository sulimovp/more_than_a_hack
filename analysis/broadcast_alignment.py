"""How well the broadcast tracking lines up with the tactical one at the offsets broadcast.offsets() finds.

    python -m analysis.broadcast_alignment

Per half: the share of frames with both balls within 3 m and their median distance at the offset, the best share
more than 50 frames away from it, and the best offset in each 3-minute window. Then whether the two metadata files
name the same team first. Aggregates only; nothing is written.
"""

import numpy as np

from possession import ball, broadcast, io, tracks

WINDOW = 3 * 60 * io.FPS
LOCAL = 40
FAR = 50


def per_half(t_ball, b_ball, half, first, stop, offset):
    frames = np.arange(first, stop)
    frames = frames[~np.isnan(t_ball[frames, 0])]
    share, n = broadcast.agreement(t_ball, b_ball, frames, offset)
    s = frames - offset
    ok = (s >= 0) & (s < len(b_ball))
    d = np.linalg.norm(t_ball[frames[ok]] - b_ball[s[ok]], axis=1)
    sample = frames[::broadcast.COARSE_SUBSAMPLE]
    far = 0.0
    for o in range(first - len(b_ball) + 1, stop, broadcast.COARSE_STEP):
        o_share, o_n = broadcast.agreement(t_ball, b_ball, sample, o)
        if abs(o - offset) > FAR and o_n * broadcast.COARSE_SUBSAMPLE >= broadcast.MIN_OVERLAP_FRAMES:
            far = max(far, o_share)
    print("half {}: offset {}  share within {:.0f} m {:.3f} over {} frames with both balls, median distance {:.2f} m;"
          " best share more than {} frames away {:.3f}".format(half, offset, broadcast.BALL_AGREE_M, share, n,
                                                              np.nanmedian(d), FAR, far))
    best = []
    for lo in range(first, stop, WINDOW):
        w = frames[(frames >= lo) & (frames < lo + WINDOW)]
        shares = [broadcast.agreement(t_ball, b_ball, w, o)[0] for o in range(offset - LOCAL, offset + LOCAL + 1)]
        best.append(offset - LOCAL + int(np.argmax(shares)))
    best = np.array(best) - offset
    print("   best offset per 3-minute window, minus the half's: min {} max {}, {} of {} windows within 3 frames".format(
        best.min(), best.max(), int(np.sum(np.abs(best) <= 3)), len(best)))


def team_convention():
    """Share of frames with a player within 1.5 m of the ball in both views on which they name the same nearest team,
    on all such frames and on those where the two balls are within 3 m."""
    t_ball, t_players, t_team = io.load_atd()
    t_players = tracks.repair_keepers(t_players, t_team)
    b_ball, b_players, b_team = broadcast.on_tactical()
    b_players = tracks.repair_keepers(b_players, b_team)
    th, ta, td = ball.nearest_distances(t_ball, t_players, t_team)
    bh, ba, bd = ball.nearest_distances(b_ball, b_players, b_team)
    with np.errstate(invalid="ignore"):
        same_ball = np.linalg.norm(t_ball - b_ball, axis=1) < broadcast.BALL_AGREE_M
        both = (td < 1.5) & (bd < 1.5) & (th != ta) & (bh != ba)
    for name, m in (("all", both), ("balls within 3 m", both & same_ball)):
        agree = np.mean((th[m] < ta[m]) == (bh[m] < ba[m]))
        print("same nearest team in both views, first metadata team as home in both ({}): {:.1%} of {} frames".format(
            name, agree, int(m.sum())))


def main():
    t_ball = io.load_atd()[0]
    b_ball = broadcast.load_atd()[0]
    shift = broadcast.offsets()
    for half, first, stop in io.halves():
        per_half(t_ball, b_ball, half, first, stop, shift[half])
    team_convention()


if __name__ == "__main__":
    main()
