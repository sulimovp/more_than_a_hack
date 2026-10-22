"""S4: which team has the ball, from the shape of both teams. Used where the ball is missing.

Self-supervised: the training labels are frames where the ball is visible, in play, and at a
player's feet (within 1 m, moving with the player). No hand labels. Validated across halves:
train on one half, test on the other.

Features are oriented by attack direction so "forward" means towards the opponent's goal for each
team, which lets one model serve both halves.
"""

import warnings

import numpy as np

from . import control
from .io import AWAY, FPS, HOME, N_FRAMES, PITCH, halves, home_attacks_right

LABEL_MAX_M = 1.0
LABEL_MAX_REL = 3.0
# Frames inside a gap are strongly correlated, and a wrong flip inside a true possession splits it
# into three. At 0.05 a 5 s gap at P = 0.8 carries about 9 nats, less than the two switches (12)
# that flipping and flipping back would cost, so shape decides handovers the geometry left open
# rather than overruling continuity.
GAIN = 0.05
SEED = 7


def _team_features(xy, vxy, forward):
    """Per-frame summary of one team. xy, vxy: (n, k, 2); forward: +1 or -1 along x."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = xy[..., 0] * forward
        vx = vxy[..., 0] * forward
        return np.stack([
            np.nanmean(x, axis=1), np.nanmean(xy[..., 1], axis=1),
            np.nanstd(x, axis=1), np.nanstd(xy[..., 1], axis=1),
            np.nanmax(x, axis=1), np.nanmin(x, axis=1),
            np.nanmean(vx, axis=1), np.nanmedian(np.linalg.norm(vxy, axis=2), axis=1),
            np.sum(x > (PITCH[0] / 2) * forward, axis=1),
        ], axis=1)


def features(players, team):
    vel = control.player_velocities(players)
    # smooth velocities over half a second; single-frame tracking jitter is large
    from scipy.ndimage import uniform_filter1d
    vel = uniform_filter1d(np.nan_to_num(vel), FPS // 2, axis=0)
    out = np.full((N_FRAMES, 21), np.nan, dtype=np.float32)
    for half, first, stop in halves():
        sl = slice(first, stop)
        fwd_home = 1.0 if home_attacks_right(half) else -1.0
        h = _team_features(players[sl][:, team == HOME], vel[sl][:, team == HOME], fwd_home)
        a = _team_features(players[sl][:, team == AWAY], vel[sl][:, team == AWAY], -fwd_home)
        gap = h[:, :1] + a[:, :1]  # centroid separation along the attack axis
        out[sl] = np.concatenate([h, a, h[:, 6:7] - a[:, 6:7], h[:, 2:3] - a[:, 2:3], gap], axis=1)
    return out


def labels(b, players, team, live_codes):
    """HOME / AWAY where a player clearly has the ball, -1 elsewhere."""
    y = np.full(N_FRAMES, -1, dtype=np.int8)
    for side in (HOME, AWAY):
        w = control.team_weight(b, players, team, side)
        # team_weight is exp(-(d/1 m)^2) times a co-movement gate; 0.37 is d = 1 m
        y[(w >= np.exp(-(LABEL_MAX_M / control.CONTACT_SCALE_M) ** 2)) & (y == -1)] = side
    y[~b["visible"] | b["oob"] | (live_codes > AWAY)] = -1
    return y


def fit_predict(X, y):
    """Cross-half: each half is predicted by a model trained on the other. Returns P(home)."""
    import lightgbm as lgb
    p = np.full(N_FRAMES, np.nan)
    report = {}
    blocks = halves()
    for i, (half, first, stop) in enumerate(blocks):
        _, o_first, o_stop = blocks[1 - i]
        train = np.zeros(N_FRAMES, dtype=bool)
        train[o_first:o_stop] = True
        train &= y >= 0
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=200, subsample=0.5, subsample_freq=1,
                                   random_state=SEED, deterministic=True, n_jobs=4, verbose=-1)
        model.fit(X[train], (y[train] == HOME).astype(int))
        p[first:stop] = model.predict_proba(X[first:stop])[:, 1]
        test = np.zeros(N_FRAMES, dtype=bool)
        test[first:stop] = True
        test &= y >= 0
        report[half] = float(np.mean((p[test] > 0.5) == (y[test] == HOME)))
    return p, report


def evidence(b, players, team, live_codes):
    """Team log-odds from shape on frames where the ball is missing; zero elsewhere."""
    X = features(players, team)
    y = labels(b, players, team, live_codes)
    p, report = fit_predict(X, y)
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-3, 1 - 1e-3)
    score = GAIN * np.log(p / (1 - p))
    score[b["visible"]] = 0.0
    return score, p, report
