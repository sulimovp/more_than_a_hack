"""S1: ball track repair and kinematics.

The automatic ball track has three kinds of error that matter here:
  - stuck detections: the ball sits still for seconds with nobody near it (a false positive that
    the tracker keeps re-confirming), which would hand possession to whoever walks past;
  - teleports: single-frame jumps faster than any kick;
  - short dropouts, under a second, inside otherwise continuous play.
Long gaps (5 s and up) are left alone: on this match they are mostly the tracker withholding the
ball during stoppages, and S2 treats them as evidence rather than as holes to fill.
"""

import numpy as np
from scipy.signal import savgol_filter

from .io import FPS, PITCH

STUCK_STILL_M = 0.3        # metres moved over one second that counts as "not moving"
STUCK_NOBODY_M = 5.0       # nobody within this distance
STUCK_MIN_FRAMES = 2 * FPS
TELEPORT_MPS = 40.0        # faster than any kick
FILL_MAX_FRAMES = FPS      # interpolate dropouts up to one second
OOB_MARGIN_M = 0.5         # past the line by this much, to absorb tracking noise
OOB_MIN_FRAMES = 2
SG_WINDOW, SG_ORDER = 7, 2


def runs(mask):
    """[(start, stop)] of True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def nearest_distances(ball, players, team):
    """(d_home, d_away, d_any) per frame, metres, NaN where the ball or a team is missing."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    with np.errstate(invalid="ignore"):
        home = np.where(team == 0, d, np.inf).min(axis=1)
        away = np.where(team == 1, d, np.inf).min(axis=1)
    d_home = np.where(np.isfinite(home), home, np.nan)
    d_away = np.where(np.isfinite(away), away, np.nan)
    return d_home, d_away, np.fmin(d_home, d_away)


def remove_stuck(ball, d_any):
    moved = np.full(len(ball), np.nan)
    moved[FPS:] = np.linalg.norm(ball[FPS:] - ball[:-FPS], axis=1)
    with np.errstate(invalid="ignore"):
        candidate = (moved < STUCK_STILL_M) & (d_any > STUCK_NOBODY_M)
    out = ball.copy()
    removed = 0
    for start, stop in runs(candidate):
        if stop - start >= STUCK_MIN_FRAMES:
            out[max(start - FPS, 0):stop] = np.nan  # the window looked back one second
            removed += stop - start + FPS
    return out, removed


def remove_teleports(ball):
    out = ball.copy()
    step = np.full(len(ball), np.nan)
    step[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=1) * FPS
    with np.errstate(invalid="ignore"):
        jumps = np.flatnonzero(step > TELEPORT_MPS)
    removed = 0
    for i in jumps:
        # an outlier jumps out and back; a real fast ball does not come straight back
        for back in range(1, 4):
            j = i + back
            if j < len(ball) and np.linalg.norm(ball[j] - ball[i - 1]) * FPS / (back + 1) < TELEPORT_MPS:
                out[i:j] = np.nan
                removed += j - i
                break
    return out, removed


def fill_short_gaps(ball):
    out = ball.copy()
    missing = np.isnan(out[:, 0])
    filled = 0
    for start, stop in runs(missing):
        if start == 0 or stop == len(out) or stop - start > FILL_MAX_FRAMES:
            continue
        t = np.arange(1, stop - start + 1) / float(stop - start + 1)
        out[start:stop] = out[start - 1] + t[:, None] * (out[stop] - out[start - 1])
        filled += stop - start
    return out, filled


def kinematics(ball):
    """Savitzky-Golay velocity (m/s) and acceleration (m/s^2) on each visible run, NaN elsewhere."""
    vel = np.full_like(ball, np.nan, dtype=np.float64)
    acc = np.full_like(ball, np.nan, dtype=np.float64)
    for start, stop in runs(~np.isnan(ball[:, 0])):
        if stop - start < SG_WINDOW:
            continue
        seg = ball[start:stop].astype(np.float64)
        vel[start:stop] = savgol_filter(seg, SG_WINDOW, SG_ORDER, deriv=1, axis=0) * FPS
        acc[start:stop] = savgol_filter(seg, SG_WINDOW, SG_ORDER, deriv=2, axis=0) * FPS * FPS
    return vel, acc


def out_of_pitch(ball):
    """Boolean per frame, and which line: 0 none, 1 left goal line, 2 right goal line, 3 touchline."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        left = x < -OOB_MARGIN_M
        right = x > PITCH[0] + OOB_MARGIN_M
        side = (y < -OOB_MARGIN_M) | (y > PITCH[1] + OOB_MARGIN_M)
    raw = left | right | side
    out = np.zeros(len(ball), dtype=bool)
    for start, stop in runs(raw):
        if stop - start >= OOB_MIN_FRAMES:
            out[start:stop] = True
    line = np.zeros(len(ball), dtype=np.int8)
    line[out & left] = 1
    line[out & right] = 2
    line[out & side & ~left & ~right] = 3
    return out, line


def repair(ball, players, team):
    """Clean ball track plus everything later stages read from it."""
    _, _, d_any = nearest_distances(ball, players, team)
    clean, n_stuck = remove_stuck(ball, d_any)
    clean, n_teleport = remove_teleports(clean)
    raw_visible = ~np.isnan(clean[:, 0])
    clean, n_filled = fill_short_gaps(clean)
    vel, acc = kinematics(clean)
    oob, oob_line = out_of_pitch(clean)
    d_home, d_away, d_any = nearest_distances(clean, players, team)
    return {
        "ball": clean,
        "visible": ~np.isnan(clean[:, 0]),
        "raw_visible": raw_visible,
        "vel": vel,
        "speed": np.linalg.norm(vel, axis=1),
        "acc": np.linalg.norm(acc, axis=1),
        "oob": oob,
        "oob_line": oob_line,
        "d_home": d_home,
        "d_away": d_away,
        "d_any": d_any,
        "report": {"stuck_frames": n_stuck, "teleport_frames": n_teleport, "filled_frames": n_filled},
    }
