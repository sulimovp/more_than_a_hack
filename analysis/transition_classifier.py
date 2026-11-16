"""Can the pipeline's own signals tell a real predicted transition from an invented one?

    python -m analysis.transition_classifier

Validation study on the hand-labelled transitions (tools/review_web.py). Each labelled transition of
the submitted file gets a handful of features read from the pipeline at that frame and over the
spell it opens; the target is "invented" (answer n) against "real" (y or e). A small regularised
logistic regression is scored by repeated stratified cross-validation (AUC). Prints the AUC per
feature set; changes nothing.
"""

import glob
import os

import numpy as np

from possession import io, pipeline
from tools.transition_review import load_answers

LABEL_DIR = os.path.join(io.DATA_DIR, "labels")
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "results",
                   "possession_pipeline.csv")
W = io.FPS * 2


def read_codes():
    import pandas as pd
    lab = pd.read_csv(CSV).label.values
    return np.array([{"H": 0, "A": 1, "D": 2}.get(x, 3) for x in lab], dtype=np.int8)


def spell_bounds(codes, f):
    a = f
    z = f
    while z < len(codes) and codes[z] == codes[f]:
        z += 1
    p = f - 1
    while p > 0 and codes[p - 1] == codes[f - 1]:
        p -= 1
    return a, z, p


def feature_rows(labels, codes, f, post):
    b = f["ball"]
    live = f["live"]
    team = f["control"] + f.get("shape", 0)
    rows, y, kinds = [], [], []
    for frame, trans, ans in labels:
        a, z, p = spell_bounds(codes, frame)
        src, dst = trans.split(">")
        kind = 0 if "D" not in (src, dst) else (1 if dst == "D" else 2)
        lo, hi = max(frame - W, 0), frame + W
        new = slice(a, z)
        lab = codes[frame]
        sgn = 1.0 if lab == io.HOME else -1.0
        d_new = b["d_home"] if dst == "H" else b["d_away"] if dst == "A" else np.minimum(b["d_home"], b["d_away"])
        rows.append([
            kind == 0, kind == 1, kind == 2,
            np.log((z - a) / io.FPS), np.log((frame - p) / io.FPS),
            np.nanmean(post[new, lab]), np.nanmin(post[lo:hi].max(axis=1)),
            np.mean(b["visible"][lo:hi]),
            np.nanmedian(np.nan_to_num(b["speed"][frame - 5:frame + 5])),
            np.nanmin(np.nan_to_num(d_new[frame - 12:frame + 13], nan=30.0)),
            np.mean(live["logodds_alive"][lo:hi]), np.mean(live["logodds_alive"][new]) if z > a else 0.0,
            np.mean(live["speed"][lo:hi]),
            np.max(f["whistle"][lo:hi]),
            np.mean(sgn * team[new]) if kind == 0 else 0.0,
        ])
        y.append(ans == "N")
        kinds.append(kind)
    names = ["switch", "to_dead", "from_dead", "log_len_new", "log_len_prev", "post_new", "post_min",
             "ball_visible", "ball_speed", "dist_new_team", "alive_near", "alive_new", "player_speed",
             "whistle", "team_evidence_new"]
    return np.nan_to_num(np.array(rows, dtype=float)), np.array(y), np.array(kinds), names


def cv_auc(X, y, reps=20, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for r in range(reps):
        pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000)).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, pred))
    return float(np.mean(aucs)), float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95))


def main():
    labels = []
    for path in glob.glob(os.path.join(LABEL_DIR, "transitions_*.csv")):
        labels += [(fr, t, a) for (fr, t), (a, _) in load_answers(path).items() if a in "YEN"]
    codes = read_codes()
    f = pipeline.features()
    first = pipeline.decode(f)
    team = pipeline.add_shape(f, first)
    post = pipeline.posteriors(f, team)
    X, y, kinds, names = feature_rows(labels, codes, f, post)
    print("{} labelled transitions, {} invented".format(len(y), int(y.sum())))
    print("all features: AUC {:.3f} (5-95% over CV repeats {:.3f}-{:.3f})".format(*cv_auc(X, y)))
    for k, kname in enumerate(["team switches", "play stops", "restarts"]):
        m = kinds == k
        if m.sum() >= 20 and 5 <= y[m].sum() <= m.sum() - 5:
            print("{}: n {} invented {}  AUC {:.3f} ({:.3f}-{:.3f})".format(kname, m.sum(), int(y[m].sum()), *cv_auc(X[m], y[m])))
    from sklearn.metrics import roc_auc_score
    print("single features (AUC of the raw value; 0.5 = none, far from 0.5 either way = signal):")
    for j, n in enumerate(names):
        if np.std(X[:, j]) > 0:
            print("  {:18s} {:.3f}".format(n, roc_auc_score(y, X[:, j])))


if __name__ == "__main__":
    main()
