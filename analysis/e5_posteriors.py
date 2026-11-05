"""E5 step 1: forward-backward posteriors per predicted spell (ROADMAP E5).

    python -m analysis.e5_posteriors

Decodes as the submission does, runs decoder.posteriors() on the final pass, and writes one row per predicted spell
(half, start, stop, label, length, mean and minimum posterior of its own label, labels either side) to
data/analysis/e5_spell_posteriors.csv. That table is frame-level match output, so it stays under data/ (gitignored);
analysis/e5_posteriors.md gets the aggregate tables and the public-window spells.

The last table is a mechanics check of E5 step 2 with the placeholder calibration P(real) = mean posterior: how many
possessions each threshold would remove, and whether the public window loses a matched possession. It is not a
selection; the threshold and the calibration come from the E1 labels (analysis/e5_fit_min_risk.py).
"""

import os

import numpy as np

from evaluation import possession_metrics as pm
from possession import decoder, io, make_submission, pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(io.DATA_DIR, "analysis")
TABLE = os.path.join(OUT_DIR, "e5_spell_posteriors.csv")
SUMMARY = os.path.join(HERE, "e5_posteriors.md")
LENGTH_BINS = (0, 1, 2, 5, 10, 20, np.inf)
POST_BINS = (0, 0.9, 0.95, 0.98, 0.99, 0.999, 1.000001)
THRESHOLDS = (0.8, 0.9, 0.95, 0.98)
LETTER = "HAD"


def final_pass(f=None):
    """(features, team evidence, codes) exactly as pipeline.run() decodes them."""
    f = pipeline.features() if f is None else f
    first = pipeline.decode(f)
    team = pipeline.add_shape(f, first)
    return f, team, pipeline.decode(f, team)


def spell_table(codes, post):
    """[dict] one per predicted run in the graded blocks, with its posterior summary."""
    rows = []
    for half, first, stop in io.halves():
        segs = decoder.spell_posteriors(codes[first:stop], post[first:stop])
        for i, (a, z, c, mean_p, min_p) in enumerate(segs):
            rows.append({"half": half, "start": first + a, "stop": first + z, "label": LETTER[c],
                         "length_s": (z - a) / float(io.FPS), "mean_post": mean_p, "min_post": min_p,
                         "prev": LETTER[segs[i - 1][2]] if i > 0 else "",
                         "next": LETTER[segs[i + 1][2]] if i + 1 < len(segs) else ""})
    return rows


def write_table(rows, path=TABLE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["half", "start", "stop", "label", "length_s", "mean_post", "min_post", "prev", "next"]
    with open(path, "w", newline="\n") as handle:
        handle.write(",".join(keys) + "\n")
        for r in rows:
            handle.write("{},{},{},{},{:.2f},{:.6f},{:.6f},{},{}\n".format(*[r[k] for k in keys]))


def crosstab(rows, labels):
    sel = [r for r in rows if r["label"] in labels]
    head = "| length | n | " + " | ".join("{}-{}".format(lo, hi if hi < 1 else "1") for lo, hi in
                                           zip(POST_BINS[:-1], POST_BINS[1:])) + " |"
    lines = [head, "|---|---|" + "---|" * (len(POST_BINS) - 1)]
    for lo, hi in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
        group = [r for r in sel if lo <= r["length_s"] < hi]
        counts = np.histogram([r["mean_post"] for r in group], bins=POST_BINS)[0]
        name = "{}-{} s".format(lo, hi) if np.isfinite(hi) else "{} s and over".format(lo)
        lines.append("| {} | {} | {} |".format(name, len(group), " | ".join(str(c) for c in counts)))
    return lines


def best_iou(gt_spells, a, z, c):
    best = 0.0
    for ga, gz, gc in gt_spells:
        if gc == c and ga < z and a < gz:
            best = max(best, (min(z, gz) - max(a, ga)) / float(max(z, gz) - min(a, ga)))
    return best


def matched_public(gt, codes):
    """Indices of the public ground-truth possessions matched at IoU 0.5, and public spellF1/macroF1."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(codes, runs)
    gt_sp, pr_sp = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), 0.5)
    res = pm.evaluate(gt, codes, bootstrap_n=0)
    return set(np.flatnonzero(flags).tolist()), res["spell_f1_mean"], res["macro_f1"]


def mechanics(f, team, codes, post, gt):
    """Removal at a few thresholds with the placeholder calibration (not a selection)."""
    base_matched, base_spell, base_macro = matched_public(gt, codes)
    blocks = [(a, z) for _, a, z in io.halves()]
    n_base = sum(1 for s in pm.segments(codes, blocks) if s[2] < 2)
    lines = ["| threshold | possessions removed | possessions (full) | D share | public spellF1 | public macroF1 | public matched (IoU 0.5) lost | sha256 |",
             "|---|---|---|---|---|---|---|---|",
             "| off | 0 | {} | {:.1%} | {:.3f} | {:.3f} | - | {} |".format(
                 n_base, np.mean(codes[io.graded_mask()] == io.DEAD), base_spell, base_macro,
                 make_submission.codes_sha256(codes)[:12])]
    for threshold in THRESHOLDS:
        out = codes.copy()
        for a, z in blocks:
            out[a:z] = decoder.remove_uncertain_spells(codes[a:z], post[a:z], threshold)
        n = sum(1 for s in pm.segments(out, blocks) if s[2] < 2)
        matched, spell, macro = matched_public(gt, out)
        lost = sorted(base_matched - matched)
        lines.append("| {} | {} | {} | {:.1%} | {:.3f} | {:.3f} | {} | {} |".format(
            threshold, n_base - n, n, np.mean(out[io.graded_mask()] == io.DEAD), spell, macro,
            ", ".join(str(i) for i in lost) if lost else "none", make_submission.codes_sha256(out)[:12]))
    return lines


def main():
    f, team, codes = final_pass()
    post = pipeline.posteriors(f, team)
    rows = spell_table(codes, post)
    write_table(rows)
    graded = io.graded_mask()
    agree = float(np.mean(np.argmax(post[graded], axis=1) == codes[graded]))
    top = post[graded].max(axis=1)
    poss = [r for r in rows if r["label"] in "HA"]
    gt = io.load_gt_public()
    gt_sp = pm.possession_spells(gt, pm.valid_runs(gt))
    w0, w1 = pm.valid_runs(gt)[0]

    out = ["# E5 step 1: posteriors per predicted spell",
           "",
           "Generated by `python -m analysis.e5_posteriors` on the submission's decode (final pass, shape evidence in).",
           "Forward-backward on the decoder's own chain: same states, emissions and transition scores, log-sum-exp in",
           "place of max (`decoder.posteriors`). The per-spell table, one row per predicted spell, is written to",
           "`data/analysis/e5_spell_posteriors.csv` (gitignored: it is the submission in another form).",
           "",
           "- graded frames where the posterior's most likely label is the Viterbi label: {:.2%}".format(agree),
           "- graded frames whose largest posterior is under 0.9: {:.2%}; under 0.6: {:.2%}".format(
               np.mean(top < 0.9), np.mean(top < 0.6)),
           "- possessions {}, stoppages {}; possession mean posterior, 5th / 25th / 50th percentile: {}".format(
               len(poss), len(rows) - len(poss),
               " / ".join("{:.3f}".format(v) for v in np.percentile([r["mean_post"] for r in poss], [5, 25, 50]))),
           "",
           "The posteriors are overconfident: every frame's evidence is counted as independent, and the gains were set",
           "for a decoder that only needs the order of the scores. Short possessions sit lower ({}), but on the public".format(
               ", ".join("median {:.3f} {}".format(np.median([r["mean_post"] for r in poss if lo <= r["length_s"] < hi]),
                                                    name) for lo, hi, name in ((0, 1, "under 1 s"), (20, np.inf, "over 20 s")))),
           "window the predicted possessions that match no true one (best IoU under 0.3) have {}. Whether the posterior".format(
               ", ".join("{:.3f} ({} {}-{})".format(r["mean_post"], r["label"], r["start"], r["stop"]) for r in poss
                         if w0 <= r["start"] < w1 and best_iou(gt_sp, max(r["start"], w0), min(r["stop"], w1),
                                                                "HAD".index(r["label"])) < 0.3) or "none"),
           "separates real from false spells, and what 0.95 means as P(the spell is real), the E1 labels have to measure.",
           "",
           "## Possessions by length and mean posterior (counts)",
           ""] + crosstab(rows, "HA") + [
           "",
           "## Stoppages by length and mean posterior (counts)",
           ""] + crosstab(rows, "D") + [
           "",
           "## Public window: every predicted spell",
           "",
           "| spell | frames | length s | mean posterior | min posterior | best IoU with a true spell of its label |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        if r["stop"] <= w0 or r["start"] >= w1:
            continue
        a, z = max(r["start"], w0), min(r["stop"], w1)
        iou = best_iou(gt_sp, a, z, "HAD".index(r["label"])) if r["label"] in "HA" else float("nan")
        out.append("| {} | {}-{} | {:.1f} | {:.3f} | {:.3f} | {} |".format(
            r["label"], a, z, (z - a) / 25.0, r["mean_post"], r["min_post"],
            "{:.2f}".format(iou) if r["label"] in "HA" else "stoppage"))
    out += ["",
            "## Mechanics check of step 2 with the placeholder calibration (not a selection)",
            "",
            "`decoder.remove_uncertain_spells` with P(real) = mean posterior. The threshold and the calibration are",
            "chosen on the E1 labels only (`python -m analysis.e5_fit_min_risk`); until then `decoder.MIN_RISK` is None",
            "and the submission is unchanged. Lost public possessions are indices into the 17 true possessions.",
            ""] + mechanics(f, team, codes, post, gt)
    with open(SUMMARY, "w", newline="\n") as handle:
        handle.write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwrote", TABLE, "and", SUMMARY)


if __name__ == "__main__":
    main()
