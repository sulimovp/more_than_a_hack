"""Run every stage and return per-frame codes for the whole video. make_submission.py writes them."""

import numpy as np

from . import audio, ball, broadcast, control, decoder, io, livemodel, liveness, restarts, shape, tracks

SECOND_HALF_KICKOFF = 113742  # FAQ.md: 110808..113741 are graded and dead
# E6: None keeps the rule-based S2. "replace", "replace+laws" or "add" bring in the weak-label LightGBM
# (livemodel.combine). Off: every mode fails the label-free checks, D share 57-60% (experiments.md, 5 Nov).
LIVENESS_MODEL = None


def features(liveness_model=None):
    """Everything the decoder reads, computed once. Separate so analysis can reuse it.
    liveness_model overrides LIVENESS_MODEL (False forces the rule-based S2)."""
    b_raw, players, team = io.load_atd()
    players = tracks.repair_keepers(players, team)
    b = ball.repair(b_raw, players, team)
    bc_ball, bc_players, bc_team = broadcast.on_tactical()
    second = {"ball": bc_ball, "players": tracks.repair_keepers(bc_players, bc_team), "team": bc_team}
    tags = io.load_tags()
    whistle = audio.whistle_score()
    live = liveness.evidence(b, players, whistle, tags)
    mode = LIVENESS_MODEL if liveness_model is None else liveness_model
    if mode:
        p, model_report = livemodel.evidence(b, players, team, whistle, tags, live, SECOND_HALF_KICKOFF)
        live = livemodel.combine(live, p, mode, b)
        live["model_report"] = model_report
    return {
        "ball": b,
        "players": players,
        "team": team,
        "tags": tags,
        "whistle": whistle,
        "live": live,
        "broadcast": second,
        "control": control.evidence(b, players, team, second),
        "restart_team": restarts.evidence(live, b, tags),
    }


def _blocks(f, team_evidence=None):
    """[(first, stop, decoder inputs in viterbi() argument order)] for each graded block."""
    force_dead = np.zeros(io.N_FRAMES, dtype=bool)
    force_dead[io.halves()[1][1]:SECOND_HALF_KICKOFF] = True
    live = f["live"]
    team = f["control"] if team_evidence is None else team_evidence
    out = []
    for _, first, stop in io.halves():
        sl = slice(first, stop)
        out.append((first, stop, (live["logodds_alive"][sl], team[sl], live["enter_dead"][sl],
                                  live["leave_dead"][sl], f["restart_team"][sl], force_dead[sl])))
    return out


def decode(f, team_evidence=None, min_risk=None, **decoder_params):
    """Viterbi codes for the whole video. min_risk is the E5 step-2 threshold; None is off."""
    codes = np.full(io.N_FRAMES, io.ABSTAIN, dtype=np.int8)
    durations = {k: v for k, v in decoder_params.items() if k in ("min_live", "min_dead")}
    for first, stop, args in _blocks(f, team_evidence):
        codes[first:stop] = decoder.viterbi(*args, **decoder_params)
        if min_risk is not None:
            post = decoder.posteriors(*args, **decoder_params)
            codes[first:stop] = decoder.remove_uncertain_spells(codes[first:stop], post, min_risk, **durations)
    return codes


def posteriors(f, team_evidence=None, **decoder_params):
    """(N_FRAMES, 3) P(H), P(A), P(D) by forward-backward on the decode() chain; NaN off the blocks."""
    post = np.full((io.N_FRAMES, 3), np.nan)
    for first, stop, args in _blocks(f, team_evidence):
        post[first:stop] = decoder.posteriors(*args, **decoder_params)
    return post


def add_shape(f, first_pass):
    """Second-pass team evidence: geometry where the ball is seen, team shape where it is not.
    The shape model's training labels need to know which frames are live, hence two passes."""
    score, _, report = shape.evidence(f["ball"], f["players"], f["team"], first_pass)
    f["shape"] = score
    f["shape_report"] = report
    return f["control"] + score


def run(use_shape=True, min_risk=None, **decoder_params):
    """Codes for the whole video. min_risk (E5 step 2) acts on the final pass only; None reads
    decoder.MIN_RISK, which stays off until the E1 labels set it."""
    f = features()
    codes = decode(f, **decoder_params)
    threshold = decoder.MIN_RISK if min_risk is None else min_risk
    if use_shape:
        codes = decode(f, add_shape(f, codes), min_risk=threshold, **decoder_params)
    elif threshold is not None:
        codes = decode(f, min_risk=threshold, **decoder_params)
    return codes
