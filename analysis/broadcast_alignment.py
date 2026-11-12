"""How well the broadcast tracking lines up with the tactical one at the offsets broadcast.offsets() finds.

    python -m analysis.broadcast_alignment

Per half: the share of frames with both balls within 3 m and their median distance at the offset, the best share
more than 50 frames away from it, and the best offset in each 3-minute window. Then whether the two metadata files
name the same team first, and what the broadcast says on the public window. Aggregates only; nothing is written.
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


def nearest(b_ball, players, team):
    """(team of the nearest player, its distance) per frame; distance NaN without a ball or players."""
    d_home, d_away, d_any = ball.nearest_distances(b_ball, players, team)
    side = np.where(np.nan_to_num(d_home, nan=np.inf) < np.nan_to_num(d_away, nan=np.inf), io.HOME, io.AWAY)
    return side, d_any


def two_views():
    """Tactical (repaired ball, repaired keepers) and broadcast (keepers repaired) nearest team and distance."""
    t_raw, t_players, t_team = io.load_atd()
    t_players = tracks.repair_keepers(t_players, t_team)
    t_ball = ball.repair(t_raw, t_players, t_team)["ball"]
    b_ball, b_players, b_team = broadcast.on_tactical()
    b_players = tracks.repair_keepers(b_players, b_team)
    with np.errstate(invalid="ignore"):
        same_ball = np.linalg.norm(t_ball - b_ball, axis=1) < broadcast.BALL_AGREE_M
    return nearest(t_ball, t_players, t_team), nearest(b_ball, b_players, b_team), same_ball, ~np.isnan(b_ball[:, 0])


def team_convention(t, b, same_ball):
    """Share of frames with a player within 1.5 m of the ball in both views on which they name the same nearest team."""
    with np.errstate(invalid="ignore"):
        both = (t[1] < 1.5) & (b[1] < 1.5)
    for name, m in (("all", both), ("balls within 3 m", both & same_ball)):
        print("same nearest team in both views, first metadata team as home in both ({}): {:.1%} of {} frames".format(
            name, np.mean(t[0][m] == b[0][m]), int(m.sum())))


def public_window(t, b, b_visible):
    """What the broadcast adds on the public window: ball visibility by true label, and nearest-team accuracy."""
    gt = io.load_gt_public()
    live, dead = gt < io.DEAD, gt == io.DEAD
    print("public window: broadcast ball visible on {:.1%} of true-live and {:.1%} of true-dead frames".format(
        b_visible[live].mean(), b_visible[dead].mean()))
    with np.errstate(invalid="ignore"):
        m = live & (t[1] < 1.0) & (b[1] < 1.0)
        tactical = live & (t[1] < 1.0)
    agree = m & (t[0] == b[0])
    print("   true-live frames with a player within 1 m in both views: {}; the views agree on {:.1%};"
          " when they agree the truth matches on {:.1%}, the tactical view alone (player within 1 m) on {:.1%};"
          " when they disagree the tactical view is right on {:.1%}".format(
              int(m.sum()), agree.sum() / m.sum(), np.mean(t[0][agree] == gt[agree]),
              np.mean(t[0][tactical] == gt[tactical]), np.mean(t[0][m & ~agree] == gt[m & ~agree])))


def main():
    t_ball = io.load_atd()[0]
    b_ball = broadcast.load_atd()[0]
    shift = broadcast.offsets()
    for half, first, stop in io.halves():
        per_half(t_ball, b_ball, half, first, stop, shift[half])
    t, b, same_ball, b_visible = two_views()
    team_convention(t, b, same_ball)
    public_window(t, b, b_visible)


if __name__ == "__main__":
    main()
