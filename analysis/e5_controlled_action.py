"""E5 sweep: credit a touch only when it is the new team's first controlled action (analysis/gt_convention.md).

    python -m analysis.e5_controlled_action

A touch by side s is dropped when the next touch within W frames is by the other side: a block, a deflection or a
challenge that does not keep the ball (the b class of E2). The pipeline keeps W = control.CONFIRM_FRAMES (50); W = 0
is the pipeline before E5. Prints the public scores and the full-match possession count per W; nothing is written.
"""
from evaluation import possession_metrics as pm
from possession import control, io, pipeline
from possession.report import public_unit_tests

def run(f, window):
    default = control.confirmed.__defaults__
    control.confirmed.__defaults__ = (window,)
    f["control"] = control.evidence(f["ball"], f["players"], f["team"], f["broadcast"])
    control.confirmed.__defaults__ = default
    first = pipeline.decode(f)
    return pipeline.decode(f, pipeline.add_shape(f, first))


def main():
    f = pipeline.features()
    gt = io.load_gt_public()
    for window in (0, 25, 44, 50, 62):
        codes = run(f, window)
        res = pm.evaluate(gt, codes, bootstrap_n=0)
        runs = [(a, b) for _, a, b in io.halves()]
        n_poss = sum(1 for s in pm.segments(codes, runs) if s[2] in (io.HOME, io.AWAY))
        print("W {:2d}: spellF1 {:.4f}  macroF1 {:.4f}  spells {} (truth {})  possessions {}".format(
            window, res["spell_f1_mean"], res["macro_f1"], res["n_pred_spells"], res["n_gt_spells"], n_poss),
            flush=True)
        if window in (0, 50):
            public_unit_tests(codes)


if __name__ == "__main__":
    main()
