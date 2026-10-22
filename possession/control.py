"""S3: per-frame evidence for which team is on the ball, from player-ball geometry.

Positive values favour home, negative favour away, zero means no information (ball missing, out
of pitch, or in flight with nobody near). The ground truth switches team when a player gains
control, so the evidence is per frame and undelayed; the decoder's switch cost and minimum spell
length are what stop a ball flying past a defender from becoming a possession.

Contact weight is a soft possession zone (Vidal-Codina et al. 2022 use 1.5 m), gated by how well
the player and the ball move together. On the public window the nearest player's team matches the
truth 90% of the time when their relative speed is under 5 m/s, and 58% above 8 m/s: a ball
passing a player fast is a deflection or a miss, not control.
"""

import numpy as np

from .io import FPS

CONTACT_SCALE_M = 1.0       # kernel width; weight 0.37 at 1 m, 0.1 at 1.5 m
EVIDENCE_GAIN = 3.0
CONTROL_REL_MPS = 5.0       # relative speed above which contact counts for little
FAST_WEIGHT = 0.3


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


def evidence(b, players, team):
    score = EVIDENCE_GAIN * (team_weight(b, players, team, 0) - team_weight(b, players, team, 1))
    score[~b["visible"] | b["oob"]] = 0.0
    return np.nan_to_num(score, nan=0.0)
