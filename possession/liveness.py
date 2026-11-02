"""S2: is the ball in play? Per-frame log-odds for the decoder, plus entry/exit event strengths.

What the data says (PLAN.md section 2): the automatic tracker withholds the ball for most of every
stoppage, and players walk during stoppages. So:
  - inside a long ball gap, the gap's median collective speed decides alive vs dead;
  - with the ball visible, collective speed decides weakly, and out-of-pitch decides strongly;
  - entering a stoppage is marked by whistles, the ball leaving the pitch, or the ball vanishing;
  - leaving one is marked by the ball reappearing or being struck from a standstill.

Emissions are log-odds per frame. Event strengths are log-bonuses the decoder adds to the
live->dead and dead->live transitions at that frame. All constants are physical guesses fixed
before looking at the public window; PLAN.md section 5 says how they may be retuned.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import ball as ballmod
from .io import FPS

SPEED_PIVOT = 1.25          # m/s collective speed separating walking (dead) from play (live)
VISIBLE_GAIN, VISIBLE_CLIP = 3.0, (-0.5, 2.5)
ACTIVITY_WINDOW = 2 * FPS   # a strike within +/- 2 s
ACTIVITY_GAIN = 0.5
GAP_PIVOT, GAP_GAIN, GAP_CLIP = 1.4, 6.0, (-3.0, 3.0)
GAP_LOCAL_GAIN = 1.5
GAP_SETPIECE = -1.0         # a set piece tagged inside the gap or just after it
OOB_LOGODDS = -4.0
SETPIECE_AFTER = 10 * FPS

# A moving ball is still in play when the players have already slowed down. The bonus is
# capped so a one-second kick inside a real stoppage cannot pay for leaving and re-entering it.
BALL_PIVOT = 5.0            # m/s
BALL_GAIN = 0.4
BALL_CLIP = 0.6
BALL_WINDOW = FPS

WHISTLE_DB = 13.0           # about the 99th percentile of the graded score
WHISTLE_BONUS = 4.0
WHISTLE_LOOKBACK = int(1.5 * FPS)
OOB_ONSET_BONUS = 4.0
GAP_START_BONUS = 3.0
GAP_END_BONUS = 3.0
STRIKE_BONUS = 3.0
STRIKE_MPS = 5.0
RESTART_LAG = 25            # frames after a strike or reappearance that the restart may land on
PENDING_LOGODDS = -3.0      # ball went out and has not been played again
PENDING_MAX = 60 * FPS


def collective_speed(players):
    """Median player speed (m/s) per frame from a centred 2-frame difference, gaps filled."""
    speed = np.full(len(players), np.nan)
    step = np.linalg.norm(players[2:] - players[:-2], axis=2) * FPS / 2.0
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            speed[1:-1] = np.nanmedian(step, axis=1)
    fill = np.nanmedian(speed)
    return np.where(np.isnan(speed), fill, speed)


def gap_table(visible, speed, tag_starts):
    """[(start, stop, median speed, set piece in or just after)] for every missing-ball run."""
    out = []
    for start, stop in ballmod.runs(~visible):
        near = bool(np.any((tag_starts >= start) & (tag_starts <= stop + SETPIECE_AFTER)))
        out.append((start, stop, float(np.median(speed[start:stop])), near))
    return out


def strike_frames(b):
    """Frames where the ball speed crosses STRIKE_MPS upwards inside the pitch.

    In the public window the ground truth turns live 0-25 frames after a restart kick takes the
    ball past about 5 m/s (free kick 16400 on the frame, goal kick 18364 -> 18389), or about 10
    frames after the ball reappears already moving (12980 -> 12989, 17826 -> 17838).
    """
    speed = np.nan_to_num(b["speed"], nan=0.0)
    fast = (speed > STRIKE_MPS) & ~b["oob"]
    return np.flatnonzero(fast[1:] & ~fast[:-1]) + 1


def pending_restart(b, strikes):
    """True from each out-of-pitch onset until the first strike after the ball is back in."""
    pending = np.zeros(len(b["oob"]), dtype=bool)
    oob_on = np.flatnonzero(np.diff(b["oob"].astype(np.int8)) == 1) + 1
    for onset in oob_on:
        back_in = onset + int(np.argmax(~b["oob"][onset:onset + PENDING_MAX]))
        later = strikes[strikes > back_in]
        end = min(later[0] if len(later) else onset + PENDING_MAX, onset + PENDING_MAX)
        pending[onset:end] = True
    return pending


def smear(n, frames, bonus, width):
    """Add bonus on [frame, frame + width) for each frame, without double counting overlaps."""
    out = np.zeros(n)
    for f in frames:
        out[f:f + width] = np.maximum(out[f:f + width], bonus)
    return out


def evidence(b, players, whistle, tags):
    """dict of per-frame arrays: logodds_alive, enter_dead, leave_dead, plus diagnostics."""
    n = len(players)
    speed = collective_speed(players)
    s2 = uniform_filter1d(speed, 4 * FPS + 1)
    visible = b["visible"]
    tag_starts = np.array([t["start_frame"] for t in tags if t["code"] in ("SET PIECES", "GOALS")])

    strikes = strike_frames(b)
    alive = np.clip(VISIBLE_GAIN * (s2 - SPEED_PIVOT), *VISIBLE_CLIP)
    # live play keeps passing the ball even when players walk; a dead ball mostly sits still
    passing = smear(n, np.maximum(strikes - ACTIVITY_WINDOW, 0), 1.0, 2 * ACTIVITY_WINDOW)
    alive += ACTIVITY_GAIN * (2.0 * passing - 1.0)
    gaps = gap_table(visible, speed, tag_starts)
    for start, stop, gap_speed, setpiece in gaps:
        score = np.clip(GAP_GAIN * (gap_speed - GAP_PIVOT), *GAP_CLIP)
        score += GAP_LOCAL_GAIN * np.clip(s2[start:stop] - SPEED_PIVOT, -1.0, 1.0)
        if setpiece:
            score += GAP_SETPIECE
        alive[start:stop] = score
    ball_speed = uniform_filter1d(np.nan_to_num(b["speed"], nan=0.0), BALL_WINDOW)
    moving = np.clip(BALL_GAIN * (ball_speed - BALL_PIVOT), 0.0, BALL_CLIP)
    moving[~visible | b["oob"]] = 0.0
    alive = alive + moving
    pending = pending_restart(b, strikes)
    alive[pending] = np.minimum(alive[pending], PENDING_LOGODDS)
    alive[b["oob"]] = OOB_LOGODDS

    enter = np.zeros(n)
    loud = np.nan_to_num(whistle, nan=0.0) > WHISTLE_DB
    # a whistle in the last WHISTLE_LOOKBACK frames (the peaks sit 0-1.4 s before a stoppage onset)
    counts = np.concatenate(([0.0], np.cumsum(loud)))
    recent_whistle = (counts[1:] - counts[np.maximum(np.arange(n) - WHISTLE_LOOKBACK, 0)]) > 0
    enter += WHISTLE_BONUS * recent_whistle
    oob_on = np.flatnonzero(np.diff(b["oob"].astype(np.int8)) == 1) + 1
    enter[oob_on] += OOB_ONSET_BONUS
    long_gaps = [(start, stop) for start, stop, _, _ in gaps if stop - start >= 2 * FPS]
    enter[[start for start, _ in long_gaps]] += GAP_START_BONUS
    leave = smear(n, [stop for _, stop in long_gaps if stop < n], GAP_END_BONUS, RESTART_LAG)
    leave = np.maximum(leave, smear(n, strikes, STRIKE_BONUS, RESTART_LAG))

    return {"logodds_alive": alive, "enter_dead": enter, "leave_dead": leave,
            "speed": speed, "speed_2s": s2, "gaps": gaps, "strikes": strikes,
            "pending": pending}
