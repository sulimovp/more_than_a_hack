"""Diagnostics for a set of codes: public-window unit tests and full-match weak checks.

    python -m possession.report [path/to/submission.csv]

The public window is 17 possessions: these numbers check that the pipeline is right, they do not
rank versions of it (EVALUATION.md, PLAN.md section 5).
"""

import sys

import numpy as np

from . import io

import possession_metrics as pm  # noqa: E402  (io put evaluation/ on the path)

POSSESSION_PHASES = {"PROGRESSION": 1, "BUILD UP": 1, "LONG BALL": 1, "ATTACKING TRANSITION": 1,
                     "DEFENSIVE TRANSITION": -1, "DEFENDING IN MIDDLE THIRD": -1,
                     "DEFENDING IN DEFENSIVE THIRD": -1, "DEFENDING IN ATTACKING THIRD": -1}
TEAM_CODE = {"Barcelona": io.HOME, "Atletic Club": io.AWAY}


def public_unit_tests(codes, out=print):
    gt = io.load_gt_public()
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(codes, runs)
    res = pm.evaluate(gt, codes, bootstrap_n=0)
    out("public window: spellF1 {:.4f}  macroF1 {:.4f}  spells {} (truth {})".format(
        res["spell_f1_mean"], res["macro_f1"], res["n_pred_spells"], res["n_gt_spells"]))
    gt_segs = pm.segments(gt, runs)
    pred_segs = pm.segments(filled, runs)
    for a, b, k in gt_segs:
        best, best_seg = 0.0, None
        for c, d, j in pred_segs:
            if j != k or d <= a or b <= c:
                continue
            iou = (min(b, d) - max(a, c)) / float(max(b, d) - min(a, c))
            if iou > best:
                best, best_seg = iou, (c, d)
        mark = "ok " if best >= 0.5 else ("~  " if best >= 0.3 else "MISS")
        out("  {} {} {:6d}-{:6d} {:5.1f}s  best IoU {:.2f} {}".format(
            mark, "HAD"[k], a, b, (b - a) / 25.0, best, best_seg or ""))
    window = slice(*runs[0])
    extra = [(c, d, "HAD"[j]) for c, d, j in pred_segs if c >= runs[0][0] and d <= runs[0][1]]
    out("  predicted segments in window: {} (truth {})".format(len(extra), len(gt_segs)))
    return res


def weak_checks(codes, tags, out=print):
    runs = [(first, stop) for _, first, stop in io.halves()]
    segs = pm.segments(codes, runs)
    spells = [s for s in segs if s[2] in (io.HOME, io.AWAY)]
    graded = io.graded_mask()
    g = codes[graded]
    out("full match: {} possessions, {} stoppages, D share {:.1%}, H share of live {:.1%}".format(
        len(spells), sum(1 for s in segs if s[2] == io.DEAD), np.mean(g == io.DEAD),
        np.mean(g[g != io.DEAD] == io.HOME)))
    lengths = np.array([(b - a) / 25.0 for a, b, _ in spells])
    out("  possession length s: p10 {:.1f}  median {:.1f}  p90 {:.1f};  under 1 s: {}".format(
        *np.percentile(lengths, [10, 50, 90]), int(np.sum(lengths < 1))))

    restarts = [a for a, b, k in segs if k in (io.HOME, io.AWAY)
                and a > 0 and codes[a - 1] == io.DEAD]
    restarts = np.array(restarts)
    setp = [t for t in tags if t["code"] in ("SET PIECES", "GOALS") and graded[min(t["start_frame"], io.N_FRAMES - 1)]]
    hit = 0
    for t in setp:
        s = t["start_frame"]
        if t["code"] == "GOALS":
            # a goal is followed by a stoppage, then the kickoff
            ok = np.any(codes[s:s + 30 * 25] == io.DEAD)
        else:
            ok = np.any((restarts >= s - 2 * 25) & (restarts <= s + 30 * 25))
        hit += bool(ok)
    out("  set-piece/goal tags followed within 30 s by a restart (goals: a stoppage): {}/{}".format(
        hit, len(setp)))

    agree = total = 0
    for t in tags:
        sign = POSSESSION_PHASES.get(t["code"])
        team = TEAM_CODE.get(t["tags"].get("Team"))
        if sign is None or team is None:
            continue
        expected = team if sign == 1 else 1 - team
        window = codes[t["start_frame"]:t["end_frame"]]
        live = window[(window == io.HOME) | (window == io.AWAY)]
        agree += int(np.sum(live == expected))
        total += len(live)
    out("  phase-tag team agreement on live frames: {:.1%} of {}".format(agree / max(total, 1), total))


def main():
    if len(sys.argv) > 1:
        codes = pm.label_codes(pm.read_label_csv(sys.argv[1]))
    else:
        from . import pipeline
        codes = pipeline.run()
    public_unit_tests(codes)
    weak_checks(codes, io.load_tags())


if __name__ == "__main__":
    main()
