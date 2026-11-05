"""E5 step 2: fit P(real | posterior, length) and the removal threshold on the E1 transition labels.

    python -m analysis.e5_fit_min_risk                          # reads data/labels/transitions_*.csv
    python -m analysis.e5_fit_min_risk --recall 0.7 --bootstrap 500

Changes nothing. It prints the fitted calibration, the expected spellF1 per threshold and merge rule, and the lines to
set in possession/decoder.py (P_REAL, MIN_RISK, MIN_RISK_MERGE) only when the gain on the labelled spells clears its
bootstrap band and the public window keeps every possession it matches at IoU 0.5. Without labels it says so and stops.

Labels. tools/transition_review.py stores one answer per predicted transition: y, n, t, e (with the true frame) or s.
Answers are matched by (frame, transition) to the current decode, so a review of an older file covers only the
transitions that did not move; answers that disagree between labellers are dropped. A predicted possession [a, b)
with label L is
  real      when the transition at a is y, or e with the corrected spell at IoU 0.5 or more with the predicted one,
            and the transition at b is y, t (the spell does end there, into another label) or such an e;
  not real  when the transition at a is n or t (no change there, or a change to another label), or the one at b is n;
  unknown   otherwise, and left out of the fit. A block edge counts as a confirmed transition.

Calibration: logistic regression of real on logit(mean posterior) and log(length in s), small ridge, Newton steps.

Threshold: for each candidate the removal runs on the full match. Expected spellF1 = 2 E[TP] / (n_pred + n_true),
where a surviving spell that is an original labelled spell counts its label, any other its calibrated P(real), and
n_true = E[TP with removal off] / recall (an assumption, --recall; the choice is also shown at 0.6 and 0.9). The
bootstrap resamples the labelled spells, refits the calibration and re-scores; a setting is recommended only if the
5th percentile of its gain over "off" is above zero (PLAN.md 5.4: otherwise keep the simpler version).
"""

import argparse
import glob
import os

import numpy as np

from analysis.e5_posteriors import final_pass, matched_public
from possession import decoder, io, pipeline
from tools.transition_review import load_answers

LABEL_DIR = os.path.join(io.DATA_DIR, "labels")
LETTER = "HAD"
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))
MERGES = ("previous", "posterior")
RIDGE = 1.0
IOU_REAL = 0.5


def consensus(answer_sets):
    """{(frame, transition): (answer, true_frame)} over labellers; skips dropped, disagreements dropped."""
    votes = {}
    for answers in answer_sets:
        for key, (ans, tf) in answers.items():
            if ans != "S":
                votes.setdefault(key, []).append((ans, tf))
    out = {}
    for key, vs in votes.items():
        kinds = {v[0] for v in vs}
        if len(kinds) != 1:
            continue
        frames = [v[1] for v in vs if v[1] is not None]
        out[key] = (vs[0][0], int(round(np.mean(frames))) if frames else None)
    return out


def _edge(answers, frame, trans, opening):
    """(status, corrected frame): status 'ok', 'wrong' or 'unknown'."""
    if trans is None:
        return "ok", frame                       # block edge
    got = answers.get((frame, trans))
    if got is None:
        return "unknown", frame
    ans, tf = got
    if ans == "Y" or (ans == "T" and not opening):
        return "ok", frame
    if ans == "N" or ans == "T":
        return "wrong", frame
    if ans == "E" and tf is not None:
        return "ok", tf
    return "unknown", frame


def spell_label(a, z, c, prev, nxt, answers):
    """1 real, 0 not real, None unknown, for the possession [a, z) of label code c (module docstring)."""
    s0, a2 = _edge(answers, a, None if prev is None else LETTER[prev] + ">" + LETTER[c], True)
    s1, z2 = _edge(answers, z, None if nxt is None else LETTER[c] + ">" + LETTER[nxt], False)
    if "wrong" in (s0, s1):
        return 0
    if "unknown" in (s0, s1):
        return None
    if z2 <= a2:
        return 0
    iou = max(0, min(z, z2) - max(a, a2)) / float(max(z, z2) - min(a, a2))
    return int(iou >= IOU_REAL)


def labelled_spells(codes, post, answers):
    """{(start, stop, code): (label, mean posterior, length s)} for possessions with a known label."""
    out = {}
    for _, first, stop in io.halves():
        segs = decoder.runs(codes[first:stop])
        for i, (a, z, c) in enumerate(segs):
            if c == io.DEAD:
                continue
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            y = spell_label(first + a, first + z, c, prev, nxt, answers)
            if y is not None:
                out[(first + a, first + z, c)] = (y, float(post[first + a:first + z, c].mean()), (z - a) / 25.0)
    return out


def fit_calibration(mean_post, length_s, y, weights=None, ridge=RIDGE, iters=100):
    """(a, b, c) of P(real) = s(a + b logit(mean posterior) + c log(length s)); intercept not penalised."""
    p = np.clip(np.asarray(mean_post, dtype=float), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.ones(len(p)), np.log(p / (1 - p)), np.log(np.maximum(length_s, 1.0 / 25))])
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    pen = ridge * np.diag([0.0, 1.0, 1.0])
    beta = np.zeros(3)
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(X @ beta)))
        grad = X.T @ (w * (y - q)) - pen @ beta
        hess = X.T @ (X * (w * q * (1 - q))[:, None]) + pen + 1e-9 * np.eye(3)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return tuple(float(v) for v in beta)


def apply(codes, post, threshold, coef, merge):
    out = codes.copy()
    if threshold is None:
        return out
    for _, first, stop in io.halves():
        out[first:stop] = decoder.remove_uncertain_spells(codes[first:stop], post[first:stop], threshold,
                                                          coef=coef, merge=merge)
    return out


def expected_counts(codes, post, labelled, coef, weights=None):
    """(E[TP], n_pred) over the surviving possessions of a decode (module docstring)."""
    tp = n = 0.0
    for _, first, stop in io.halves():
        for a, z, c in decoder.runs(codes[first:stop]):
            if c == io.DEAD:
                continue
            key = (first + a, first + z, c)
            if key in labelled:
                w = 1.0 if weights is None else weights.get(key, 0.0)
                tp += w * labelled[key][0]
                n += w
            else:
                tp += float(decoder.p_real(post[key[0]:key[1], c].mean(), (z - a) / 25.0, coef))
                n += 1.0
    return tp, n


def curve(codes, post, labelled, coef, recall, weights=None):
    """{(merge, threshold or None): expected spellF1}."""
    tp0, n0 = expected_counts(codes, post, labelled, coef, weights)
    n_true = tp0 / recall
    out = {(None, None): 2 * tp0 / (n0 + n_true)}
    for merge in MERGES:
        for threshold in THRESHOLDS:
            tp, n = expected_counts(apply(codes, post, threshold, coef, merge), post, labelled, coef, weights)
            out[(merge, threshold)] = 2 * tp / (n + n_true)
    return out


def best(scores):
    """Highest expected spellF1; ties go to 'off', then to the lower threshold (fewer removals)."""
    return max(scores, key=lambda k: (round(scores[k], 9), k == (None, None), -(k[1] or 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=LABEL_DIR, help="folder with transitions_<name>.csv")
    parser.add_argument("--recall", type=float, default=0.75, help="assumed recall of true possessions")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.labels, "transitions_*.csv")))
    answers = consensus([load_answers(p) for p in files])
    if not answers:
        print("no E1 transition labels in {} yet (tools/transition_review.py); nothing to fit, "
              "decoder.MIN_RISK stays None".format(args.labels))
        return 0

    f, team, codes = final_pass()
    post = pipeline.posteriors(f, team)
    labelled = labelled_spells(codes, post, answers)
    keys = sorted(labelled)
    y = np.array([labelled[k][0] for k in keys])
    mp = np.array([labelled[k][1] for k in keys])
    ln = np.array([labelled[k][2] for k in keys])
    n_poss = sum(1 for _, a0, z0 in io.halves() for s in decoder.runs(codes[a0:z0]) if s[2] != io.DEAD)
    print("{} answers from {} file(s); {} of {} predicted possessions labelled, {} real".format(
        len(answers), len(files), len(keys), n_poss, int(y.sum())))
    if len(keys) < 20 or y.min() == y.max():
        print("too few labelled possessions, or only one class: keep decoder.MIN_RISK None")
        return 0

    coef = fit_calibration(mp, ln, y)
    print("calibration P(real) = s({:.3f} + {:.3f} logit(mean posterior) + {:.3f} log(length s))".format(*coef))
    q = decoder.p_real(mp, ln, coef)
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        sel = (q >= lo) & (q < hi)
        if sel.any():
            print("  P(real) {:.2f}-{:.2f}: n {:3d}, observed real {:.2f}".format(lo, min(hi, 1), int(sel.sum()), y[sel].mean()))

    scores = curve(codes, post, labelled, coef, args.recall)
    choice = best(scores)
    print("expected spellF1 (recall {}): off {:.4f}".format(args.recall, scores[(None, None)]))
    for merge in MERGES:
        print("  {:9s} ".format(merge) + " ".join("{}:{:.4f}".format(t, scores[(merge, t)]) for t in THRESHOLDS))
    for r in (0.6, 0.9):
        print("  best at recall {}: {}".format(r, best(curve(codes, post, labelled, coef, r))))
    print("best:", choice, "gain {:+.4f}".format(scores[choice] - scores[(None, None)]))
    if choice == (None, None):
        print("removal does not help on these labels: keep decoder.MIN_RISK None")
        return 0

    rng = np.random.default_rng(2027)
    gains = []
    for _ in range(args.bootstrap):
        draw = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)))
        weights = {k: float(w) for k, w in zip(keys, draw)}
        coef_b = fit_calibration(mp, ln, y, weights=draw)
        tp0, n0 = expected_counts(codes, post, labelled, coef_b, weights)
        n_true = tp0 / args.recall
        tp, n = expected_counts(apply(codes, post, choice[1], coef_b, choice[0]), post, labelled, coef_b, weights)
        gains.append(2 * tp / (n + n_true) - 2 * tp0 / (n0 + n_true))
    lo, hi = np.percentile(gains, [5, 95])
    print("bootstrap gain of {} over off, 5-95%: {:+.4f} to {:+.4f} ({} draws)".format(choice, lo, hi, args.bootstrap))

    gt = io.load_gt_public()
    before, s0, m0 = matched_public(gt, codes)
    after, s1, m1 = matched_public(gt, apply(codes, post, choice[1], coef, choice[0]))
    lost = sorted(before - after)
    print("public window: spellF1 {:.3f} -> {:.3f}, macroF1 {:.3f} -> {:.3f}, matched possessions lost: {}".format(
        s0, s1, m0, m1, lost or "none"))
    if lo <= 0 or lost:
        print("not recommended: {}; keep decoder.MIN_RISK None".format(
            "the gain is inside its bootstrap band" if lo <= 0 else "the public window loses a matched possession"))
        return 0
    print("recommended, set in possession/decoder.py and regenerate (make_submission, pytest, expected_sha256, "
          "experiments.md):")
    print("    P_REAL = ({:.3f}, {:.3f}, {:.3f})".format(*coef))
    print("    MIN_RISK = {}".format(choice[1]))
    print("    MIN_RISK_MERGE = \"{}\"".format(choice[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
