"""Run every stage and return per-frame codes for the whole video. make_submission.py writes them."""

import numpy as np

from . import audio, ball, control, decoder, io, liveness, restarts, shape, tracks

SECOND_HALF_KICKOFF = 113742  # FAQ.md: 110808..113741 are graded and dead


def features():
    """Everything the decoder reads, computed once. Separate so analysis can reuse it."""
    b_raw, players, team = io.load_atd()
    players = tracks.repair_keepers(players, team)
    b = ball.repair(b_raw, players, team)
    tags = io.load_tags()
    whistle = audio.whistle_score()
    live = liveness.evidence(b, players, whistle, tags)
    return {
        "ball": b,
        "players": players,
        "team": team,
        "tags": tags,
        "whistle": whistle,
        "live": live,
        "control": control.evidence(b, players, team),
        "restart_team": restarts.evidence(live, b, tags),
    }


def decode(f, team_evidence=None, **decoder_params):
    codes = np.full(io.N_FRAMES, io.ABSTAIN, dtype=np.int8)
    force_dead = np.zeros(io.N_FRAMES, dtype=bool)
    force_dead[io.halves()[1][1]:SECOND_HALF_KICKOFF] = True
    live = f["live"]
    team = f["control"] if team_evidence is None else team_evidence
    for _, first, stop in io.halves():
        sl = slice(first, stop)
        codes[sl] = decoder.viterbi(
            live["logodds_alive"][sl], team[sl], live["enter_dead"][sl],
            live["leave_dead"][sl], f["restart_team"][sl], force_dead[sl], **decoder_params)
    return codes


def add_shape(f, first_pass):
    """Second-pass team evidence: geometry where the ball is seen, team shape where it is not.
    The shape model's training labels need to know which frames are live, hence two passes."""
    score, _, report = shape.evidence(f["ball"], f["players"], f["team"], first_pass)
    f["shape"] = score
    f["shape_report"] = report
    return f["control"] + score


def run(use_shape=True, **decoder_params):
    f = features()
    codes = decode(f, **decoder_params)
    if use_shape:
        codes = decode(f, add_shape(f, codes), **decoder_params)
    return codes
