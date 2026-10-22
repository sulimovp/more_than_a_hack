"""S5: which team restarts play, as log-odds the decoder applies on dead -> live transitions.

Sources, strongest first:
  - kickoffs, from the laws: the second half is kicked off by the side that did not kick off the
    first, and after a goal the conceding side kicks off (Smart Tagging GOALS names the scorer);
  - Smart Tagging set pieces with a team (corners, goal kicks, free kicks);
  - otherwise (every throw-in is tagged N/A) the team of the player nearest the ball when it is
    struck or reappears, who is the taker.
Values are applied over the same window as the dead -> live bonus (liveness.RESTART_LAG).
"""

import numpy as np

from .io import AWAY, FPS, HOME, N_FRAMES, halves
from .liveness import RESTART_LAG

TEAM = {"Barcelona": HOME, "Atletic Club": AWAY}
TAG_LOOKBACK = 20 * FPS     # a set-piece tag starts 3-12 s before the kick in the public window
TAG_LOOKAHEAD = 2 * FPS
GOAL_LOOKBACK = 120 * FPS
KICKOFF_WINDOW = 3 * FPS
TAG_LOGODDS = 4.0
KICKOFF_LOGODDS = 6.0
TAKER_LOGODDS = 2.0
TAKER_MAX_M = 2.0
SIGN = {HOME: 1.0, AWAY: -1.0}


def candidates(live):
    """Frames where play may restart: the start of each window with a dead -> live bonus."""
    on = live["leave_dead"] > 0
    return np.flatnonzero(on & ~np.concatenate(([False], on[:-1])))


def first_kickoff_team(gt_first_label=AWAY):
    """Who kicked off the first half. Read off the video (Athletic, frame 11185); the public
    ground truth's first label agrees. The second half goes to the other side."""
    return gt_first_label


def evidence(live, b, tags):
    out = np.zeros(N_FRAMES)
    starts = candidates(live)
    setpieces = [t for t in tags if t["code"] == "SET PIECES"]
    goals = [t for t in tags if t["code"] == "GOALS" and t["tags"].get("Team") in TEAM]
    second_half_kickoff = None
    for t in setpieces:
        if t["tags"].get("Type") == "Kick Off (Start)" and t["start_frame"] > halves()[1][1] - 10 * FPS:
            second_half_kickoff = t["start_frame"]

    for s in starts:
        value = 0.0
        # kickoffs
        if second_half_kickoff is not None and abs(s - second_half_kickoff) < 20 * FPS:
            value = KICKOFF_LOGODDS * SIGN[1 - first_kickoff_team()]
        after_goal = [g for g in goals if 0 < s - g["start_frame"] < GOAL_LOOKBACK]
        kickoff_tags = [t for t in setpieces if t["tags"].get("Type") == "Kick Off (After Goal)"
                        and -KICKOFF_WINDOW * 4 < s - t["start_frame"] < TAG_LOOKBACK]
        if after_goal and kickoff_tags:
            scorer = TEAM[after_goal[-1]["tags"]["Team"]]
            value = KICKOFF_LOGODDS * SIGN[1 - scorer]
        # tagged set pieces with a team
        if value == 0.0:
            near = [t for t in setpieces if t["tags"].get("Team") in TEAM
                    and -TAG_LOOKAHEAD <= s - t["start_frame"] <= TAG_LOOKBACK]
            if near:
                value = TAG_LOGODDS * SIGN[TEAM[near[-1]["tags"]["Team"]]]
        # the taker
        if value == 0.0:
            window = slice(s, min(s + RESTART_LAG, N_FRAMES))
            d_home = np.nanmin(np.append(b["d_home"][window], np.inf))
            d_away = np.nanmin(np.append(b["d_away"][window], np.inf))
            if min(d_home, d_away) < TAKER_MAX_M:
                value = TAKER_LOGODDS * (1.0 if d_home < d_away else -1.0)
        out[s:s + RESTART_LAG] = value
    return out
