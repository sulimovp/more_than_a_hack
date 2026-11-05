"""S2 v2 (ROADMAP E6): P(ball in play) from a small LightGBM trained on weak labels, cross-half.

Features per frame, after Lang et al. (2022): collective speed and collective acceleration over 0.4, 1, 2, 4 and
10 s; ball visible; time elapsed and remaining in the current ball gap; ball out of pitch; ball distance to the
nearest line; ball speed; whistle score, max over 1 s; time to the next and since the last set-piece or goal tag;
players in the fuller and in the emptier penalty box; share of players standing in their own half.

Weak labels on the full match, no hand label: dead = ball out of pitch, the 50 frames before a kickoff, ball gaps
whose median collective speed is under 1.0 m/s; live = ball visible and over 5 m/s with collective speed over 2 m/s.
Where both apply, dead wins. About 39% of graded frames get a label.

Each half is predicted by a model trained on the other (as shape.py), and P(alive) is rounded to 0.02 so that a
last-bit difference in LightGBM across machines cannot change the file. pipeline.LIVENESS_MODEL chooses how the
result reaches the decoder; it is off by default (experiments.md, 5 Nov, has why).
"""

import warnings

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from . import ball as ballmod
from .io import FPS, HOME, N_FRAMES, PITCH, halves, home_attacks_right
from .liveness import OOB_LOGODDS, PENDING_LOGODDS

WINDOWS_S = (0.4, 1.0, 2.0, 4.0, 10.0)
TAG_CAP_S = 120.0
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
KICKOFF_DEAD = 50                 # frames before a kickoff labelled dead
KICKOFF_OFFSET = 85               # tag start to kick: 88 and 82 frames at the two half-start kickoffs
GAP_DEAD_MPS = 1.0
LIVE_BALL_MPS = 5.0
LIVE_PLAYERS_MPS = 2.0
SEED = 7
QUANTUM = 0.02
GAIN = 1.0
FEATURES = (["speed_{}s".format(w) for w in WINDOWS_S] + ["accel_{}s".format(w) for w in WINDOWS_S]
            + ["visible", "gap_elapsed_s", "gap_remaining_s", "oob", "line_m", "ball_mps", "whistle_1s",
               "to_tag_s", "since_tag_s", "box_max", "box_min", "own_half"])
BALL_FEATURES = ("visible", "gap_elapsed_s", "gap_remaining_s", "oob", "line_m", "ball_mps")


def _shift(x, k):
    """x[t + k] with the edge value repeated."""
    out = np.empty_like(x)
    if k >= 0:
        out[:len(x) - k] = x[k:]
        out[len(x) - k:] = x[-1]
    else:
        out[-k:] = x[:k]
        out[:-k] = x[0]
    return out


def tag_starts(tags):
    return np.sort(np.array([t["start_frame"] for t in tags if t["code"] in ("SET PIECES", "GOALS")], dtype=int))


def kickoff_frames(tags, strikes, second_half_kickoff):
    """Kick frames: 11185 (the first graded frame), the second-half kickoff, and after each goal the first
    strike inside the kickoff tag (tag start + KICKOFF_OFFSET when no strike is seen)."""
    out = [halves()[0][1], second_half_kickoff]
    for t in tags:
        if t["code"] == "SET PIECES" and t["tags"].get("Type") == "Kick Off (After Goal)":
            inside = strikes[(strikes >= t["start_frame"]) & (strikes <= t["end_frame"])]
            out.append(int(inside[0]) if len(inside) else t["start_frame"] + KICKOFF_OFFSET)
    return sorted(out)


def features(b, players, team, whistle, tags, speed):
    """(N_FRAMES, len(FEATURES)) float32; NaN where a feature is undefined (LightGBM handles it)."""
    X = np.full((N_FRAMES, len(FEATURES)), np.nan, dtype=np.float32)
    smooth = uniform_filter1d(speed, int(WINDOWS_S[0] * FPS))
    for i, w in enumerate(WINDOWS_S):
        size = int(round(w * FPS))
        X[:, i] = uniform_filter1d(speed, size)
        h = max(size // 2, 1)
        X[:, len(WINDOWS_S) + i] = (_shift(smooth, h) - _shift(smooth, -h)) * FPS / (2.0 * h)
    col = len(WINDOWS_S) * 2
    visible = b["visible"]
    X[:, col] = visible
    elapsed = np.zeros(N_FRAMES)
    remaining = np.zeros(N_FRAMES)
    for start, stop in ballmod.runs(~visible):
        idx = np.arange(start, stop)
        elapsed[idx] = (idx - start) / FPS
        remaining[idx] = (stop - idx) / FPS
    X[:, col + 1], X[:, col + 2] = elapsed, remaining
    X[:, col + 3] = b["oob"]
    x, y = b["ball"][:, 0], b["ball"][:, 1]
    X[:, col + 4] = np.minimum(np.minimum(x, PITCH[0] - x), np.minimum(y, PITCH[1] - y))
    X[:, col + 5] = b["speed"]
    X[:, col + 6] = maximum_filter1d(np.nan_to_num(whistle, nan=0.0), FPS)
    starts = tag_starts(tags)
    frames = np.arange(N_FRAMES)
    nxt = np.searchsorted(starts, frames, side="left")
    to_tag = np.where(nxt < len(starts), (starts[np.minimum(nxt, len(starts) - 1)] - frames) / FPS, TAG_CAP_S)
    prv = np.searchsorted(starts, frames, side="right") - 1
    since = np.where(prv >= 0, (frames - starts[np.maximum(prv, 0)]) / FPS, TAG_CAP_S)
    X[:, col + 7] = np.minimum(to_tag, TAG_CAP_S)
    X[:, col + 8] = np.minimum(since, TAG_CAP_S)
    px, py = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):
        wide = np.abs(py - PITCH[1] / 2.0) < BOX_HALF_WIDTH
        left = np.sum(wide & (px < BOX_DEPTH), axis=1)
        right = np.sum(wide & (px > PITCH[0] - BOX_DEPTH), axis=1)
    X[:, col + 9] = np.maximum(left, right)
    X[:, col + 10] = np.minimum(left, right)
    own = np.full(N_FRAMES, np.nan)
    for half, first, stop in halves():
        home_left = home_attacks_right(half)
        sl = slice(first, stop)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            in_left = px[sl] < PITCH[0] / 2.0
            mine = np.where(team == HOME, in_left == home_left, in_left != home_left)
            tracked = ~np.isnan(px[sl])
            own[sl] = np.sum(mine & tracked, axis=1) / np.maximum(np.sum(tracked, axis=1), 1)
    X[:, col + 11] = own
    return X


def weak_labels(b, speed, gaps, kickoffs):
    """1 live, 0 dead, -1 no label (module docstring)."""
    y = np.full(N_FRAMES, -1, dtype=np.int8)
    ball_speed = np.nan_to_num(b["speed"], nan=0.0)
    y[b["visible"] & (ball_speed > LIVE_BALL_MPS) & (speed > LIVE_PLAYERS_MPS)] = 1
    for start, stop, median_speed, _ in gaps:
        if median_speed < GAP_DEAD_MPS:
            y[start:stop] = 0
    for k in kickoffs:
        y[max(k - KICKOFF_DEAD, 0):k] = 0
    y[b["oob"]] = 0
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, first, stop in halves():
        graded[first:stop] = True
    y[~graded] = -1
    return y


def fit_predict(X, y):
    """Cross-half: each half is predicted by a model trained on the other. Returns P(alive) and a report."""
    import lightgbm as lgb
    p = np.full(N_FRAMES, np.nan)
    report = {"importance": np.zeros(X.shape[1])}
    blocks = halves()
    for i, (half, first, stop) in enumerate(blocks):
        _, o_first, o_stop = blocks[1 - i]
        train = np.zeros(N_FRAMES, dtype=bool)
        train[o_first:o_stop] = True
        train &= y >= 0
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=200, subsample=0.5, subsample_freq=1,
                                   random_state=SEED, deterministic=True, n_jobs=4, verbose=-1)
        model.fit(X[train], y[train].astype(int))
        p[first:stop] = model.predict_proba(X[first:stop])[:, 1]
        test = np.zeros(N_FRAMES, dtype=bool)
        test[first:stop] = True
        labelled = test & (y >= 0)
        report[half] = {"accuracy": float(np.mean((p[labelled] > 0.5) == (y[labelled] == 1))),
                        "labelled": int(labelled.sum()),
                        "alive_unlabelled": float(np.mean(p[test & (y < 0)] > 0.5))}
        report["importance"] += model.booster_.feature_importance(importance_type="gain")
    return p, report


def evidence(b, players, team, whistle, tags, live, second_half_kickoff, drop=()):
    """(P(alive) rounded to QUANTUM, report) over the whole video; 0.5 outside the graded blocks.

    drop names features to leave out. BALL_FEATURES is the diagnostic of 5 Nov: every weak-labelled live frame has
    a visible ball and every labelled gap frame is dead, so with the ball features in, the model learns "ball
    missing or outside means dead" (line_m, NaN when the ball is missing, takes 95% of the gain)."""
    speed = live["speed"]
    X = features(b, players, team, whistle, tags, speed)
    keep = [i for i, name in enumerate(FEATURES) if name not in drop]
    X = X[:, keep]
    y = weak_labels(b, speed, live["gaps"], kickoff_frames(tags, live["strikes"], second_half_kickoff))
    p, report = fit_predict(X, y)
    report["features"] = [FEATURES[i] for i in keep]
    report["label_share"] = {"live": float(np.mean(y == 1)), "dead": float(np.mean(y == 0))}
    p = np.clip(np.round(np.nan_to_num(p, nan=0.5) / QUANTUM) * QUANTUM, 1e-3, 1 - 1e-3)
    return p, report


def combine(live, p, mode, b):
    """A copy of the rule-based liveness dict with its alive log-odds taken from the model.

    "replace": GAIN logit(p), nothing else. "replace+laws": the same, then the two laws of the rule-based
    S2 on top (a ball out of pitch is dead, and so is the stretch until it is struck back in).
    "add": the rule-based log-odds plus GAIN logit(p)."""
    out = dict(live)
    model = GAIN * np.log(p / (1.0 - p))
    if mode == "add":
        out["logodds_alive"] = live["logodds_alive"] + model
    elif mode in ("replace", "replace+laws"):
        alive = model.copy()
        if mode == "replace+laws":
            pending = live["pending"]
            alive[pending] = np.minimum(alive[pending], PENDING_LOGODDS)
            alive[b["oob"]] = OOB_LOGODDS
        out["logodds_alive"] = alive
    else:
        raise ValueError("unknown liveness model mode {!r}".format(mode))
    out["p_model"] = p
    return out
