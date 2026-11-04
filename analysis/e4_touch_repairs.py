"""E4 sweep: three touch-detector repairs, one at a time and stacked (ROADMAP E4).

    python -m analysis.e4_touch_repairs

  base     the pipeline as it is (PRE_PEAK_FRAMES 0, TOUCH_ACC 20, ORPHAN_ACC 0)
  pre5     deflection tested on the mean relative speed over the 5 frames before the peak
  pre5+15  as pre5, TOUCH_ACC 15
  orphan   an acceleration peak over 15 m/s^2 with nobody within 1.5 m goes to the side of a slot that was within
           3 m in the last second and is missing now
  reacq    (added after the first sweep) the first visible frame after a gap of 3+ frames is a touch by the nearest
           player when within 0.5 or 1.0 m, whatever the relative speed (the ball reappears as it leaves the foot)
Prints the public scores, the full-match possession count and the public unit-test table around the named cases
(15346 for 1 and 2, 17293 for 3); nothing is written.
"""
from evaluation import possession_metrics as pm
from possession import control, io, pipeline
from possession.report import public_unit_tests

VARIANTS = {
    "base": {},
    "pre5": {"PRE_PEAK_FRAMES": 5},
    "pre5+15": {"PRE_PEAK_FRAMES": 5, "TOUCH_ACC": 15.0},
    "orphan": {"ORPHAN_ACC": 15.0},
    "reacq.5": {"REACQUIRE_M": 0.5},
    "reacq1": {"REACQUIRE_M": 1.0},
}


def run(f, settings):
    old = {k: getattr(control, k) for k in settings}
    for k, v in settings.items():
        setattr(control, k, v)
    try:
        f["control"] = control.evidence(f["ball"], f["players"], f["team"])
        first = pipeline.decode(f)
        return pipeline.decode(f, pipeline.add_shape(f, first))
    finally:
        for k, v in old.items():
            setattr(control, k, v)


def main():
    f = pipeline.features()
    gt = io.load_gt_public()
    runs = [(a, b) for _, a, b in io.halves()]
    for name, settings in VARIANTS.items():
        codes = run(f, settings)
        res = pm.evaluate(gt, codes, bootstrap_n=0)
        n_poss = sum(1 for s in pm.segments(codes, runs) if s[2] in (io.HOME, io.AWAY))
        print("== {:8s} spellF1 {:.4f}  macroF1 {:.4f}  spells {} (truth {})  possessions {}".format(
            name, res["spell_f1_mean"], res["macro_f1"], res["n_pred_spells"], res["n_gt_spells"], n_poss), flush=True)
        lines = []
        public_unit_tests(codes, out=lines.append)
        for ln in lines:
            if any(t in ln for t in ("153", "172", "173", "MISS")):
                print(ln)


if __name__ == "__main__":
    main()
