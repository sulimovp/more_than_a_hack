"""E6 sweep: the weak-label liveness model (possession/livemodel.py) and decoder.viterbi(normalise_team=True).

    python -m analysis.e6_liveness_model

Variants, each decoded as the submission is (first pass, shape model, final pass):
  rule          the rule-based S2 (the submission)
  model         alive log-odds = logit P(alive) of the weak-label LightGBM
  model+laws    as model, with the rule-based laws on top: out of pitch and the pending restart are dead
  rule+model    rule-based log-odds plus the model's
each with normalise_team off and on, for two models: "spec" (all features, as ROADMAP E6 specifies) and "noball"
(the ball features left out, livemodel.BALL_FEATURES: the diagnostic for the shortcut the spec model learns).
Prints each model's cross-half report, then per variant the label-free checks
of ROADMAP E6 (D share 40-50%, set-piece restarts >= 57/62, possessions, phase agreement), the D share where the ball
is visible and where it is missing, and the public window: spellF1, macroF1 and the eight edges of its four
stoppages (predicted minus true, frames). Nothing is written.
"""

import numpy as np

from evaluation import possession_metrics as pm
from possession import io, livemodel, make_submission, pipeline, report

MODES = (None, "replace", "replace+laws", "add")
NAMES = {None: "rule", "replace": "model", "replace+laws": "model+laws", "add": "rule+model"}


def run(f, normalise):
    first = pipeline.decode(f, normalise_team=normalise)
    return pipeline.decode(f, pipeline.add_shape(f, first), normalise_team=normalise)


def stoppage_edges(gt, codes):
    """[(true start, true stop, start offset, stop offset, IoU)] for the public window's stoppages."""
    runs = pm.valid_runs(gt)
    pred = [s for s in pm.segments(pm.fill_abstentions(codes, runs), runs) if s[2] == io.DEAD]
    out = []
    for a, z, k in pm.segments(gt, runs):
        if k != io.DEAD:
            continue
        best = max(pred, key=lambda s: min(z, s[1]) - max(a, s[0]))
        inter = min(z, best[1]) - max(a, best[0])
        iou = inter / float(max(z, best[1]) - min(a, best[0])) if inter > 0 else 0.0
        out.append((a, z, best[0] - a, best[1] - z, iou))
    return out


def summary(codes, tags, gt, visible):
    lines = []
    report.weak_checks(codes, tags, out=lines.append)
    graded = io.graded_mask()
    runs = [(a, z) for _, a, z in io.halves()]
    segs = pm.segments(codes, runs)
    res = pm.evaluate(gt, codes, bootstrap_n=0)
    return {
        "spell": res["spell_f1_mean"], "macro": res["macro_f1"],
        "poss": sum(1 for s in segs if s[2] < 2), "stops": sum(1 for s in segs if s[2] == 2),
        "dshare": float(np.mean(codes[graded] == io.DEAD)),
        "d_vis": float(np.mean(codes[graded & visible] == io.DEAD)),
        "d_gap": float(np.mean(codes[graded & ~visible] == io.DEAD)),
        "setp": lines[-2].rsplit(":", 1)[1].strip(),
        "phase": float(lines[-1].split(":")[1].split("%")[0]),
        "under1": int(lines[1].rsplit(":", 1)[1]),
        "edges": stoppage_edges(gt, codes),
    }


def describe(name, p, rep, b, graded):
    print("{}: weak labels live {:.1%}, dead {:.1%} of all frames".format(
        name, rep["label_share"]["live"], rep["label_share"]["dead"]))
    for half in (1, 2):
        r = rep[half]
        print("  half {}: cross-half accuracy on {} weak-labelled frames {:.1%}; unlabelled frames called alive {:.1%}"
              .format(half, r["labelled"], r["accuracy"], r["alive_unlabelled"]))
    order = np.argsort(rep["importance"])[::-1]
    total = rep["importance"].sum()
    print("  gain importance: " + ", ".join("{} {:.0%}".format(rep["features"][i], rep["importance"][i] / total)
                                            for i in order[:6]))
    print("  P(alive) > 0.5 on graded frames {:.1%}; ball visible {:.1%}, ball missing {:.1%}".format(
        np.mean(p[graded] > 0.5), np.mean(p[graded & b["visible"]] > 0.5), np.mean(p[graded & ~b["visible"]] > 0.5)))


def main():
    f = pipeline.features(liveness_model=False)
    rule_live = f["live"]
    b = f["ball"]
    graded = io.graded_mask()
    models = {}
    for key, drop in (("spec", ()), ("noball", livemodel.BALL_FEATURES)):
        p, rep = livemodel.evidence(b, f["players"], f["team"], f["whistle"], f["tags"], rule_live,
                                    pipeline.SECOND_HALF_KICKOFF, drop=drop)
        describe(key, p, rep, b, graded)
        models[key] = p

    gt = io.load_gt_public()
    base = None
    head = ("variant", "nt", "pub spellF1", "pub macroF1", "poss", "stops", "D share", "D vis", "D gap",
            "set pieces", "phase", "<1 s", "frames changed", "sha256")
    print("\n| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    edge_rows = []
    variants = [("rule", None, None)] + [(NAMES[m] + " " + k, m, k) for k in models for m in MODES[1:]]
    for name, mode, key in variants:
        f["live"] = rule_live if mode is None else livemodel.combine(rule_live, models[key], mode, b)
        for normalise in (False, True):
            codes = run(f, normalise)
            s = summary(codes, f["tags"], gt, b["visible"])
            if base is None:
                base = codes
            changed = int(np.sum(codes[graded] != base[graded]))
            print("| {} | {} | {:.3f} | {:.3f} | {} | {} | {:.1%} | {:.1%} | {:.1%} | {} | {:.1f}% | {} | {} | {} |".format(
                name, "on" if normalise else "off", s["spell"], s["macro"], s["poss"], s["stops"],
                s["dshare"], s["d_vis"], s["d_gap"], s["setp"], s["phase"], s["under1"], changed,
                make_submission.codes_sha256(codes)[:12]), flush=True)
            edge_rows.append((name, normalise, s["edges"]))
    f["live"] = rule_live
    print("\npublic stoppages, predicted minus true start / stop in frames (IoU):")
    print("  true: " + "  ".join("{}-{}".format(a, z) for a, z, _, _, _ in edge_rows[0][2]))
    for name, normalise, edges in edge_rows:
        print("  {:17s} nt {:3s} ".format(name, "on" if normalise else "off")
              + "  ".join("{:+5d}/{:+5d} ({:.2f})".format(ds, dz, iou) for _, _, ds, dz, iou in edges))


if __name__ == "__main__":
    main()
