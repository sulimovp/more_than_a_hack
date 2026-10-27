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
    return turn


def touch_events(b, dist, rel):
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

    frames = np.flatnonzero(candidate & b["visible"] & ~b["oob"])
    kept_f, kept_s = [], []
    last = -TOUCH_HOLD_FRAMES
    for frame in frames:
        side = int(nearer[frame])
        if dist[frame, side] > TOUCH_M:
            continue
        r = rel[frame, side]
        if np.isfinite(r) and r > DEFLECTION_MPS:
            continue
        if frame - last < TOUCH_HOLD_FRAMES:
            continue
        kept_f.append(frame)
        kept_s.append(side)
        last = frame
    return np.asarray(kept_f, dtype=int), np.asarray(kept_s, dtype=int)


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


def evidence(b, players, team):
    dist, rel = _closest(b, players, team)
    frames, sides = touch_events(b, dist, rel)
    latched = latch_signal(len(players), frames, sides, halves())
    score = EVIDENCE_GAIN * (zone_signal(b, dist, rel) + LATCH_WEIGHT * latched)
    score[~b["visible"] | b["oob"]] = 0.0
    return np.nan_to_num(score, nan=0.0)
