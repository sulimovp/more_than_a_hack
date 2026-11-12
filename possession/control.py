"""S3: per-frame evidence for which team is on the ball, from player-ball geometry.

Positive values favour home, negative favour away, zero means no information (ball missing or
out of pitch). The ground truth switches team when a player gains control, so the evidence is
undelayed; the decoder's switch cost and minimum spell length stop a ball flying past a defender
from becoming a possession.

Two signals, as in PLAN.md S3:
  - a possession zone of 1.5 m (Vidal-Codina et al. 2022), weight exp(-d / 0.7 m). Both teams
    in the zone contribute only a small nudge toward the nearer player.
  - a touch (ELASTIC: ball acceleration peaks near 20 m/s^2, or a clear change of direction,
    or the ball staying at a player's feet). The toucher's team is held from that frame until
    the next touch, which is what carries a pass while the ball is in the air.

A touch counts only as the new team's first controlled action (analysis/gt_convention.md): a touch
followed within CONFIRM_FRAMES by the other team's touch is a block, a deflection or a lost challenge
and is dropped, so the latch never flips for it.

The first visible frame after a gap of REACQUIRE_GAP frames is a touch by a player within REACQUIRE_M, whatever the
relative speed: the ball often reappears as it leaves the kicker's foot (E4, public case 17293).

With a second view (the broadcast tracking on the tactical timeline, broadcast.py), a confirmed touch is dropped when
that view sees the same ball, within VETO_BALL_M, with a player of the other team nearest it within VETO_PLAYER_M. The
second view only removes touches; it never adds one or sets a team.

A ball faster than 8 m/s with nobody within 2 m contributes no zone evidence. Relative speed
above 8 m/s is a deflection, not a touch. On the public window the nearest player's team matches
the truth 90% of the time when their relative speed is under 5 m/s, and 58% above 8 m/s.
"""

import numpy as np

from .io import FPS, halves

CONTACT_SCALE_M = 1.0       # kernel width for the shape-model labels; 0.37 at 1 m
EVIDENCE_GAIN = 3.0
CONTROL_REL_MPS = 5.0       # relative speed above which contact counts for little
FAST_WEIGHT = 0.3
ZONE_M = 1.5
ZONE_SCALE_M = 0.7          # exp(-d / 0.7); about 0.24 at 1 m, 0.12 at 1.5 m
TOUCH_ACC = 20.0            # m/s^2, ELASTIC's saturation scale for a kick or trap
TOUCH_TURN_DEG = 25.0       # a deflection-sized change of direction, not frame-to-frame noise
TOUCH_M = 1.5
TOUCH_HOLD_M = 1.0          # sustained control: the ball stays at a player's feet
TOUCH_HOLD_FRAMES = 3
DEFLECTION_MPS = 8.0
FLIGHT_MPS = 8.0
FLIGHT_CLEAR_M = 2.0
LATCH_WEIGHT = 0.5          # zone is -1..1; the latch is weaker, so a new touch can replace it
DUEL_WEIGHT = 0.35
CONFIRM_FRAMES = 50         # 2 s; public scores flat for 44-62 (experiments.md, 3 Nov E5)
PRE_PEAK_FRAMES = 0         # E4.1: >0 tests deflection on the mean relative speed over these frames before the peak
ORPHAN_ACC = 0.0            # E4.3: >0 attributes an unclaimed acceleration peak to a slot that just went missing
ORPHAN_M = 3.0
ORPHAN_FRAMES = FPS         # look back one second
REACQUIRE_M = 0.5           # E4.4 (kept 4 Nov): makes the first visible frame after a gap a touch when a player is this close
REACQUIRE_GAP = 3
VETO_BALL_M = 3.0           # the two views' balls agree: they see the same ball (broadcast.BALL_AGREE_M)
VETO_PLAYER_M = 1.5         # a second-view player this close is at the ball (TOUCH_M)


def player_velocities(players):
    vel = np.full_like(players, np.nan)
    vel[1:-1] = (players[2:] - players[:-2]) * FPS / 2.0
    return vel


def team_weight(b, players, team, side):
    """Contact weight of the closest player of one team, gated by co-movement."""
    d = np.linalg.norm(players - b["ball"][:, None, :], axis=2)
    d = np.where(np.isnan(d) | (team[None, :] != side), np.inf, d)
    j = d.argmin(axis=1)
    rows = np.arange(len(d))
    dist = d[rows, j]
    rel = np.linalg.norm(b["vel"] - player_velocities(players)[rows, j], axis=1)
    gate = np.where(np.nan_to_num(rel, nan=0.0) > CONTROL_REL_MPS, FAST_WEIGHT, 1.0)
    return np.exp(-(dist / CONTACT_SCALE_M) ** 2) * gate


def _closest(b, players, team):
    """(dist, rel_speed) for the nearest player of each team. dist is inf if that team is absent."""
    d = np.linalg.norm(players - b["ball"][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    vel_p = player_velocities(players)
    n = len(d)
    dist = np.full((n, 2), np.inf)
    rel = np.full((n, 2), np.nan)
    rows = np.arange(n)
    for side in (0, 1):
        dd = np.where(team == side, d, np.inf)
        j = dd.argmin(axis=1)
        dist[:, side] = dd[rows, j]
        delta = b["vel"] - vel_p[rows, j]
        rel[:, side] = np.linalg.norm(delta, axis=1)
    return dist, rel


def _local_peaks(values, threshold):
    peak = np.zeros(len(values), dtype=bool)
    peak[1:-1] = ((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
                  & (values[1:-1] >= threshold))
    return peak


def _direction_change(vel, speed):
    """Degrees the ball's velocity turned since the previous frame. 0 below 5 m/s."""
    v = np.nan_to_num(vel, nan=0.0)
    dot = np.sum(v[1:] * v[:-1], axis=1)
    n1 = np.linalg.norm(v[1:], axis=1)
    n0 = np.linalg.norm(v[:-1], axis=1)
    cos = np.clip(dot / np.maximum(n1 * n0, 1e-6), -1.0, 1.0)
    turn = np.zeros(len(vel))
    turn[1:] = np.degrees(np.arccos(cos))
    turn[np.nan_to_num(speed, nan=0.0) < FLIGHT_MPS * 0.6] = 0.0
    # both frames must carry a velocity: against a missing (zero) neighbour the angle is 90
    # degrees, which used to mark a touch at the first frame of every visible run
    has = ~np.isnan(vel[:, 0])
    turn[1:][~(has[1:] & has[:-1])] = 0.0
    return turn


def _deflection_speed(rel, frame, side):
    if PRE_PEAK_FRAMES == 0 or frame == 0:
        return rel[frame, side]
    window = rel[max(0, frame - PRE_PEAK_FRAMES):frame, side]
    window = window[np.isfinite(window)]
    return window.mean() if len(window) else np.nan


def orphan_sides(b, players, team, frames):
    """Side of the slot within ORPHAN_M of the ball in the last ORPHAN_FRAMES and missing now, per frame; -1 if none."""
    out = np.full(len(frames), -1, dtype=int)
    for k, frame in enumerate(frames):
        lo = max(0, frame - ORPHAN_FRAMES)
        d = np.linalg.norm(players[lo:frame] - b["ball"][lo:frame, None, :], axis=2)
        near = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=0) <= ORPHAN_M if frame > lo else np.zeros(len(team), bool)
        gone = near & np.isnan(players[frame, :, 0])
        sides = set(team[gone].tolist()) - {-1}
        if len(sides) == 1:
            out[k] = sides.pop()
    return out


def touch_events(b, dist, rel, players=None, team=None):
    """(frames, sides) of touches. side 0 is home, 1 is away.

    A touch is an acceleration peak, a direction change, or the onset of the ball staying at a
    player's feet. The player must be within TOUCH_M, and not merely have the ball fly past.
    """
    speed = np.nan_to_num(b["speed"], nan=0.0)
    acc = np.nan_to_num(b["acc"], nan=0.0)
    turn = _direction_change(b["vel"], speed)
    candidate = _local_peaks(acc, TOUCH_ACC) | _local_peaks(turn, TOUCH_TURN_DEG)

    nearer = dist.argmin(axis=1)
    hold = dist.min(axis=1) <= TOUCH_HOLD_M
    rel_near = rel[np.arange(len(rel)), nearer]
    hold &= np.isnan(rel_near) | (rel_near <= CONTROL_REL_MPS)
    # a one-frame contact is a deflection; require the ball to stay
    sustained = np.zeros(len(hold), dtype=bool)
    run = 0
    for i, flag in enumerate(hold):
        run = run + 1 if flag else 0
        if run == TOUCH_HOLD_FRAMES:
            sustained[i - TOUCH_HOLD_FRAMES + 1] = True
    candidate |= sustained
    reacquired = np.zeros(len(hold), dtype=bool)
    if REACQUIRE_M > 0:
        vis = b["visible"].astype(bool)
        gap = 0
        for i, v in enumerate(vis):
            if v and gap >= REACQUIRE_GAP and dist[i].min() <= REACQUIRE_M:
                reacquired[i] = True
            gap = 0 if v else gap + 1
    candidate |= reacquired

    frames = np.flatnonzero(candidate & b["visible"] & ~b["oob"])
    kept_f, kept_s = [], []
    last = -TOUCH_HOLD_FRAMES
    for frame in frames:
        side = int(nearer[frame])
        if dist[frame, side] > TOUCH_M:
            continue
        r = _deflection_speed(rel, frame, side)
        if np.isfinite(r) and r > DEFLECTION_MPS and not reacquired[frame]:
            continue
        if frame - last < TOUCH_HOLD_FRAMES:
            continue
        kept_f.append(frame)
        kept_s.append(side)
        last = frame
    if ORPHAN_ACC > 0 and players is not None:
        cand = np.flatnonzero(_local_peaks(acc, ORPHAN_ACC) & b["visible"] & ~b["oob"]
                              & (dist.min(axis=1) > TOUCH_M))
        cand = np.array([f for f in cand if all(abs(f - k) >= TOUCH_HOLD_FRAMES for k in kept_f)], dtype=int)
        sides = orphan_sides(b, players, team, cand)
        order = np.argsort(np.r_[kept_f, cand[sides >= 0]], kind="stable")
        kept_f = list(np.r_[kept_f, cand[sides >= 0]][order])
        kept_s = list(np.r_[kept_s, sides[sides >= 0]][order])
    return confirmed(np.asarray(kept_f, dtype=int), np.asarray(kept_s, dtype=int))


def confirmed(frames, sides, window=CONFIRM_FRAMES):
    """Drop a touch when the next one, within window frames, is by the other team."""
    if window == 0 or len(frames) < 2:
        return frames, sides
    keep = np.ones(len(frames), dtype=bool)
    keep[:-1] = ~((sides[1:] != sides[:-1]) & (np.diff(frames) <= window))
    return frames[keep], sides[keep]


def contradicted(b, frames, sides, second):
    """True for each touch the second view contradicts. second: {"ball", "players", "team"} on the same frames.

    The two balls must be within VETO_BALL_M (else the views may be looking at different things), and the second
    view's nearest player must be within VETO_PLAYER_M and of the other team. A tie in distance contradicts nothing."""
    sep = np.linalg.norm(b["ball"][frames] - second["ball"][frames], axis=1)
    d = np.linalg.norm(second["players"][frames] - second["ball"][frames][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    d_home = np.where(second["team"] == 0, d, np.inf).min(axis=1)
    d_away = np.where(second["team"] == 1, d, np.inf).min(axis=1)
    other_nearer = np.where(sides == 0, d_away < d_home, d_home < d_away)
    at_ball = np.minimum(d_home, d_away) < VETO_PLAYER_M
    return (np.nan_to_num(sep, nan=np.inf) < VETO_BALL_M) & at_ball & other_nearer


def latch_signal(n, frames, sides, blocks):
    """+/-1 from each touch until the next one, and 0 before the first touch in a block."""
    signal = np.zeros(n)
    for _, first, stop in blocks:
        val = np.zeros(stop - first)
        for frame, side in zip(frames, sides):
            if first <= frame < stop:
                val[frame - first] = 1.0 if side == 0 else -1.0
        if not np.any(val):
            continue
        changed = val != 0
        index = np.maximum.accumulate(np.where(changed, np.arange(len(val)), 0))
        filled = val[index]
        filled[:int(np.flatnonzero(changed)[0])] = 0.0
        signal[first:stop] = filled
    return signal


def zone_signal(b, dist, rel):
    """-1..1 from the possession zone. Zero while the ball is in flight with nobody near it."""
    weight = np.exp(-np.minimum(dist, 1e3) / ZONE_SCALE_M)
    weight = np.where(dist <= ZONE_M, weight, 0.0)
    fast = np.isfinite(rel) & (rel > DEFLECTION_MPS)
    weight = np.where(fast, weight * FAST_WEIGHT, weight)
    both = (weight[:, 0] > 0) & (weight[:, 1] > 0)
    diff = weight[:, 0] - weight[:, 1]
    diff = np.where(both, diff * DUEL_WEIGHT, diff)
    speed = np.nan_to_num(b["speed"], nan=0.0)
    d_any = np.minimum(dist[:, 0], dist[:, 1])
    flight = (speed > FLIGHT_MPS) & (d_any > FLIGHT_CLEAR_M)
    diff = np.where(flight, 0.0, diff)
    return diff


def evidence(b, players, team, second=None):
    """Home-minus-away control log-odds per frame. second is a second view for contradicted(), or None."""
    dist, rel = _closest(b, players, team)
    frames, sides = touch_events(b, dist, rel, players, team)
    if second is not None:
        keep = ~contradicted(b, frames, sides, second)
        frames, sides = frames[keep], sides[keep]
    latched = latch_signal(len(players), frames, sides, halves())
    score = EVIDENCE_GAIN * (zone_signal(b, dist, rel) + LATCH_WEIGHT * latched)
    score[~b["visible"] | b["oob"]] = 0.0
    return np.nan_to_num(score, nan=0.0)
