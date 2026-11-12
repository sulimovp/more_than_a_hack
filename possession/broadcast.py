"""S0b: the broadcast recording's tracking, put on the tactical timeline.

The broadcast ATD has the same format and the same 0-1 pitch coordinates as the tactical one, but
its own frame numbers. Within a half the two recordings run at the same rate, so one constant
offset per half aligns them:

    tactical video frame = broadcast row + offset[half]       (broadcast row = its ATD frame_id - 1)

The offset is not given with the data. It is found here as the shift that puts the two ball tracks
on the same spot most often (offsets()), searched over every shift that overlaps the half, and
cached. Replays and cut-aways do not move it; they only lower the share of agreeing frames.
"""

import json
import os

import numpy as np

from . import io
from games import asset_paths, source_frames  # noqa: E402  (io put the repository root on the path)

N_BROADCAST = source_frames("broadcast")
BALL_AGREE_M = 3.0          # the two ball tracks show the same ball
COARSE_STEP = 4             # offsets tried in the first pass; agreement stays high for several frames around the peak
COARSE_SUBSAMPLE = 7        # every 7th tactical frame in the first pass
REFINE_FRAMES = 2 * COARSE_STEP
MIN_OVERLAP_FRAMES = 10 * 60 * io.FPS   # both balls visible for at least 10 minutes, or the share means little


def load_atd():
    """(ball (n, 2), players (n, k, 2), team (k,)) on the broadcast's own frames, metres, NaN where missing."""
    path = io.cache_path("broadcast_atd.npz")
    if not os.path.exists(path):
        paths = asset_paths(source="broadcast")
        team = io.team_of_each_track(paths["atd_meta"])
        ball, players = io.parse_atd(paths["atd_raw"], len(team), N_BROADCAST)
        np.savez_compressed(path, ball=ball, players=players, team=team)
    data = np.load(path)
    return data["ball"], data["players"], data["team"]


def agreement(tactical_ball, broadcast_ball, frames, offset):
    """(share within BALL_AGREE_M, number of frames with both balls) over these tactical frames."""
    s = frames - offset
    ok = (s >= 0) & (s < len(broadcast_ball))
    d = np.linalg.norm(tactical_ball[frames[ok]] - broadcast_ball[s[ok]], axis=1)
    d = d[~np.isnan(d)]
    return (float(np.mean(d < BALL_AGREE_M)) if len(d) else 0.0), len(d)


def find_offset(tactical_ball, broadcast_ball, first, stop):
    """The offset with the highest ball agreement over tactical frames first..stop; ties go to the smaller offset.

    A coarse pass over every overlapping shift on a subsample, then every frame around its best."""
    frames = np.arange(first, stop)
    frames = frames[~np.isnan(tactical_ball[frames, 0])]
    sample = frames[::COARSE_SUBSAMPLE]
    coarse = np.arange(first - len(broadcast_ball) + 1, stop, COARSE_STEP)
    shares = np.zeros(len(coarse))
    for i, offset in enumerate(coarse):
        share, n = agreement(tactical_ball, broadcast_ball, sample, offset)
        if n * COARSE_SUBSAMPLE >= MIN_OVERLAP_FRAMES:
            shares[i] = share
    best = int(coarse[np.argmax(shares)])
    fine = np.arange(best - REFINE_FRAMES, best + REFINE_FRAMES + 1)
    shares = [agreement(tactical_ball, broadcast_ball, frames, offset)[0] for offset in fine]
    return int(fine[np.argmax(shares)])


def offsets():
    """{half: offset} with tactical frame = broadcast row + offset, cached in data/cache/."""
    path = io.cache_path("broadcast_offsets.json")
    if not os.path.exists(path):
        tactical_ball = io.load_atd()[0]
        broadcast_ball = load_atd()[0]
        found = {half: find_offset(tactical_ball, broadcast_ball, first, stop) for half, first, stop in io.halves()}
        with open(path, "w") as handle:
            json.dump(found, handle)
    with open(path) as handle:
        return {int(half): offset for half, offset in json.load(handle).items()}


def on_tactical():
    """(ball, players, team) of the broadcast on the tactical timeline, metres. NaN outside the graded halves
    and on tactical frames the broadcast does not cover."""
    b_ball, b_players, team = load_atd()
    ball = np.full((io.N_FRAMES, 2), np.nan, dtype=np.float32)
    players = np.full((io.N_FRAMES,) + b_players.shape[1:], np.nan, dtype=np.float32)
    shift = offsets()
    for half, first, stop in io.halves():
        t = np.arange(first, stop)
        s = t - shift[half]
        ok = (s >= 0) & (s < len(b_ball))
        ball[t[ok]] = b_ball[s[ok]]
        players[t[ok]] = b_players[s[ok]]
    return ball, players, team
