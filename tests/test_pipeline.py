"""Decoder unit cases, alignment constants, and byte-for-byte reproduction of the submission.

    python -m pytest tests -q

The reproduction test needs the match data and takes about a minute; it is skipped without it.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import control, decoder, io  # noqa: E402

EXPECTED_HASH = os.path.join(ROOT, "tests", "expected_sha256.txt")


def decode(alive, team, enter=None, leave=None, restart=None, **kw):
    n = len(alive)
    z = np.zeros(n)
    return decoder.viterbi(np.asarray(alive, float), np.asarray(team, float),
                           z if enter is None else enter, z if leave is None else leave,
                           z if restart is None else restart, np.zeros(n, bool), **kw)


def letters(codes):
    return "".join("HAD"[c] for c in codes)


def test_interception_switches_on_first_contact():
    team = [3.0] * 50 + [0.0] * 20 + [-3.0] * 50   # home, ball in flight, away player takes it
    out = letters(decode([3.0] * 120, team))
    assert out[:50] == "H" * 50 and out[70:] == "A" * 50
    assert out.index("A") >= 50  # no switch before the ball leaves the home player


def test_ball_in_flight_keeps_possession():
    team = [3.0] * 50 + [0.0] * 40 + [3.0] * 50     # pass between home players
    assert set(letters(decode([3.0] * 140, team))) == {"H"}


def test_two_frame_flicker_is_not_a_possession():
    team = [3.0] * 50 + [-3.0] * 2 + [3.0] * 50
    assert set(letters(decode([3.0] * 102, team))) == {"H"}


def test_short_duel_spell_survives():
    team = [3.0] * 50 + [-3.0] * 19 + [3.0] * 50     # 0.76 s, like the public window's shortest
    out = letters(decode([3.0] * 119, team))
    assert "A" * 15 in out


def test_stoppage_splits_same_team():
    alive = [3.0] * 60 + [-3.0] * 100 + [3.0] * 60
    team = [3.0] * 220
    out = letters(decode(alive, team))
    assert out[:60] == "H" * 60 and out[70:150] == "D" * 80 and out[-50:] == "H" * 50


def test_short_dead_dip_is_ignored():
    alive = [3.0] * 60 + [-3.0] * 20 + [3.0] * 60    # under the 2 s minimum stoppage
    assert "D" not in letters(decode(alive, [3.0] * 140))


def _ball_with_one_touch():
    """One home touch at frame 6, then the ball in flight with nobody near it."""
    n = 40
    ball = np.zeros((n, 2))
    ball[:, 0] = np.linspace(0, 40, n)
    players = np.full((n, 2, 2), np.nan)
    players[6, 0] = ball[6]
    vel = np.zeros((n, 2))
    vel[:, 0] = 12.0
    acc = np.zeros(n)
    acc[6] = 40.0
    track = {
        "ball": ball, "vel": vel, "speed": np.full(n, 12.0), "acc": acc,
        "visible": np.ones(n, dtype=bool), "oob": np.zeros(n, dtype=bool),
    }
    return track, players, np.array([0, 1])


def test_touch_is_the_player_at_the_ball():
    track, players, team = _ball_with_one_touch()
    dist, rel = control._closest(track, players, team)
    frames, sides = control.touch_events(track, dist, rel)
    assert 6 in frames
    assert sides[list(frames).index(6)] == 0


def test_latch_holds_through_a_block_and_resets():
    signal = control.latch_signal(
        20, np.array([3, 15]), np.array([0, 1]), [(1, 0, 10), (2, 10, 20)])
    assert signal[3] > 0 and signal[9] > 0
    assert signal[10] == 0 and signal[15] < 0 and signal[2] == 0


def test_flight_keeps_the_last_touch_when_the_zone_is_empty():
    track, players, team = _ball_with_one_touch()
    # the synthetic frames sit outside both halves, so score the latch on one block
    dist, rel = control._closest(track, players, team)
    frames, sides = control.touch_events(track, dist, rel)
    latched = control.latch_signal(len(track["ball"]), frames, sides, [(1, 0, 40)])
    score = control.EVIDENCE_GAIN * (control.zone_signal(track, dist, rel) + control.LATCH_WEIGHT * latched)
    assert score[6] > 0
    assert score[25] > 0          # ball is in flight and nobody is near; the touch is still held
    assert score[0] == 0


def test_second_view_vetoes_only_a_contradicting_touch():
    n = 5
    b = {"ball": np.zeros((n, 2))}
    second = {"ball": np.zeros((n, 2)), "players": np.full((n, 2, 2), np.nan), "team": np.array([0, 1])}
    second["players"][:, 1] = (1.0, 0.0)              # an away player 1 m from the broadcast ball on every frame
    second["ball"][1] = (5.0, 0.0)                    # frame 1: the views see different balls
    second["players"][1, 1] = (6.0, 0.0)
    second["players"][2, 0] = (0.5, 0.0)              # frame 2: a home player is nearer
    second["players"][3, 1] = (2.0, 0.0)              # frame 3: the away player is outside VETO_PLAYER_M
    second["ball"][4] = np.nan                        # frame 4: no broadcast ball
    frames = np.array([0, 1, 2, 3, 4])
    home = control.contradicted(b, frames, np.zeros(n, int), second)
    assert home.tolist() == [True, False, False, False, False]
    assert not control.contradicted(b, frames[:1], np.ones(1, int), second).any()   # an away touch is confirmed


def test_broadcast_offset_search_recovers_a_shift(monkeypatch):
    from possession import broadcast
    rng = np.random.default_rng(2)
    n_bc, shift = 30000, 7321
    walk = np.cumsum(rng.normal(0, 1.2, (n_bc, 2)), axis=0)        # a ball moving about 1.7 m per frame
    bc_ball = walk.astype(np.float32)
    bc_ball[rng.uniform(size=n_bc) < 0.2] = np.nan
    tactical = np.full((shift + n_bc + 5000, 2), np.nan, dtype=np.float32)
    tactical[shift:shift + n_bc] = walk + rng.normal(0, 0.5, (n_bc, 2))
    monkeypatch.setattr(broadcast, "MIN_OVERLAP_FRAMES", 5000)
    assert broadcast.find_offset(tactical, bc_ball, 0, len(tactical)) == shift


@pytest.mark.skipif(not os.path.exists(io.asset_paths(source="broadcast")["atd_raw"]), reason="broadcast data not present")
def test_broadcast_offsets_on_this_match():
    from possession import broadcast
    assert broadcast.offsets() == {1: 4014, 2: 26920}


def test_graded_blocks_and_frame_count():
    assert io.N_FRAMES == 199319
    assert io.halves() == [(1, 11185, 90652), (2, 110808, 190198)]


@pytest.mark.skipif(not os.path.exists(io.asset_paths()["atd_raw"]), reason="match data not present")
def test_submission_reproduces_byte_for_byte(tmp_path):
    from possession import make_submission
    out = tmp_path / "rerun.csv"
    assert make_submission.main(["--out", str(out)]) == 0
    with open(EXPECTED_HASH) as handle:
        expected = handle.read().split()[0]
    assert make_submission.sha256(str(out)) == expected


def test_direction_change_has_no_edge_artefact():
    vel = np.full((10, 2), np.nan)
    vel[3:8] = (6.0, 0.0)                              # a visible run entered at frame 3
    turn = control._direction_change(vel, np.linalg.norm(np.nan_to_num(vel), axis=1))
    assert turn[3] == 0.0 and turn[8] == 0.0           # no 90-degree turn against a missing frame
    vel[6] = (0.0, 6.0)
    turn = control._direction_change(vel, np.linalg.norm(np.nan_to_num(vel), axis=1))
    assert turn[6] > 80.0                              # a real turn still registers


def test_restart_holder_is_the_player_over_the_still_ball():
    from possession import restarts
    n = 200
    track = {"speed": np.full(n, 0.2), "visible": np.ones(n, dtype=bool),
             "d_home": np.full(n, 0.8), "d_away": np.full(n, 6.0)}
    assert restarts.holder(track, 150) == io.HOME
    track["speed"][:] = 9.0                           # ball never still: no answer
    assert restarts.holder(track, 150) is None


# -- E5: posteriors and minimum-risk removal ----------------------------------------------------

def _brute_force_marginals(alive, team, enter, leave, restart, force, k, m, **kw):
    """P(H), P(A), P(D) per frame by enumerating every legal path of the viterbi() chain."""
    import itertools
    e_h, e_a, e_d = decoder._emissions(alive, team, force, kw.get("normalise_team", False))
    H0, A0, D0 = 0, k, 2 * k
    Ht, At, Dt = k - 1, 2 * k - 1, 2 * k + m - 1
    base = [0] * k + [1] * k + [2] * m
    sw, l2d, d2l = kw["switch"], kw["live_to_dead"], kw["dead_to_live"]

    def trans(s0, s1, t):
        for first, tail in ((H0, Ht), (A0, At), (D0, Dt)):
            if (first <= s0 < tail and s1 == s0 + 1) or (s0 == tail and s1 == tail):
                return 0.0
        r = 0.5 * restart[t]
        table = {(At, H0): sw, (Dt, H0): d2l + leave[t] + r, (Ht, A0): sw, (Dt, A0): d2l + leave[t] - r,
                 (Ht, D0): l2d + enter[t], (At, D0): l2d + enter[t]}
        return table.get((s0, s1))

    marg = np.zeros((len(alive), 3))
    for s_init in (Ht, At, Dt):
        for path in itertools.product(range(2 * k + m), repeat=len(alive)):
            score, prev = 0.0, s_init
            for t, s in enumerate(path):
                step = trans(prev, s, t)
                if step is None:
                    break
                score += step + (e_h, e_a, e_d)[base[s]][t]
                prev = s
            else:
                for t, s in enumerate(path):
                    marg[t, base[s]] += np.exp(score)
    return marg / marg.sum(axis=1, keepdims=True)


def test_posteriors_match_brute_force_enumeration():
    rng = np.random.default_rng(0)
    n, k, m = 4, 2, 2
    args = [rng.normal(0, 2, n), rng.normal(0, 2, n), rng.normal(0, 1, n), rng.normal(0, 1, n),
            rng.normal(0, 2, n), np.array([False, False, True, False])]
    kw = dict(switch=-1.0, live_to_dead=-2.0, dead_to_live=-1.5)
    for normalise in (False, True):
        post = decoder.posteriors(*args, min_live=k, min_dead=m, normalise_team=normalise, **kw)
        exact = _brute_force_marginals(*args, k, m, normalise_team=normalise, **kw)
        assert np.allclose(post, exact, atol=1e-12)


def test_posteriors_follow_viterbi_and_spread_an_undated_switch():
    team = [3.0] * 50 + [0.0] * 20 + [-3.0] * 50       # the interception case: 20 neutral frames in flight
    alive = np.full(120, 3.0)
    z = np.zeros(120)
    post = decoder.posteriors(alive, np.asarray(team), z, z, z, np.zeros(120, bool))
    assert np.allclose(post.sum(axis=1), 1.0)
    clear = np.r_[0:50, 70:120]
    assert np.array_equal(post.argmax(axis=1)[clear], decode(alive, team)[clear])
    # Viterbi has to pick one frame for the switch; the posterior says any frame of the flight is as likely
    assert np.all(np.diff(post[50:70, 0]) < 0) and post[59, 0] > 0.5 > post[60, 0]


def _post_for(codes, p_own):
    """Posteriors that give each frame's own label p_own and split the rest evenly."""
    post = np.full((len(codes), 3), 0.0)
    post[:] = ((1 - np.asarray(p_own)) / 2.0)[:, None]
    post[np.arange(len(codes)), codes] = p_own
    return post


def test_min_risk_off_changes_nothing():
    codes = np.array([0] * 50 + [1] * 10 + [0] * 50, dtype=np.int8)
    post = _post_for(codes, np.full(110, 0.4))
    assert decoder.remove_uncertain_spells(codes, post, None) is codes
    assert decoder.MIN_RISK is None            # the submission does not use E5 step 2 until E1 sets it


def test_min_risk_drops_an_uncertain_short_spell_only():
    codes = np.array([0] * 50 + [1] * 10 + [0] * 50 + [1] * 200, dtype=np.int8)
    p = np.full(310, 0.99)
    p[50:60] = 0.6
    out = decoder.remove_uncertain_spells(codes, _post_for(codes, p), 0.7)
    assert letters(out) == "H" * 110 + "A" * 200
    # a long spell that is unsure on average is removed only if the calibration lets length count for nothing
    p[110:] = 0.6
    long_kept = decoder.remove_uncertain_spells(codes, _post_for(codes, p), 0.7, coef=(0.0, 1.0, 2.0))
    assert letters(long_kept)[110:] == "A" * 200


def test_min_risk_merge_rules():
    codes = np.array([2] * 60 + [1] * 10 + [0] * 60, dtype=np.int8)   # D, an unsure A, then H
    post = _post_for(codes, np.full(130, 0.99))
    post[60:70] = (0.0, 0.4, 0.6)                                      # the A frames look dead
    previous = decoder.remove_uncertain_spells(codes, post, 0.5, merge="previous")
    posterior = decoder.remove_uncertain_spells(codes, post, 0.5, merge="posterior")
    assert letters(previous) == "D" * 60 + "H" * 70                    # previous label is D: next team
    assert letters(posterior) == "D" * 70 + "H" * 60                   # the posterior says dead


def test_e5_spell_labels_from_transition_answers():
    from analysis import e5_fit_min_risk as fit
    answers = fit.consensus([{(100, "A>H"): ("Y", None), (200, "H>A"): ("T", None), (300, "A>H"): ("N", None)},
                             {(100, "A>H"): ("Y", None), (300, "A>H"): ("Y", None)}])
    assert (300, "A>H") not in answers                                 # labellers disagree: dropped
    assert fit.spell_label(100, 200, 0, 1, 1, answers) == 1           # opening y, closing t: it ends there
    answers[(100, "A>H")] = ("N", None)
    assert fit.spell_label(100, 200, 0, 1, 1, answers) == 0           # no change at the opening
    answers[(100, "A>H")] = ("E", 150)
    assert fit.spell_label(100, 200, 0, 1, 1, answers) == 1           # half a second late on a 4 s spell
    answers[(100, "A>H")] = ("E", 180)
    assert fit.spell_label(100, 200, 0, 1, 1, answers) == 0           # IoU 0.2
    assert fit.spell_label(100, 400, 0, 1, 1, answers) is None        # closing transition unanswered
    assert fit.spell_label(0, 200, 0, None, 1, answers) == 1          # a block edge counts as confirmed


def test_e5_calibration_fit_finds_the_direction():
    from analysis import e5_fit_min_risk as fit
    rng = np.random.default_rng(1)
    mean_post = rng.uniform(0.6, 0.999, 400)
    length = rng.uniform(0.5, 30.0, 400)
    z = -2.0 + 1.0 * np.log(mean_post / (1 - mean_post)) + 0.5 * np.log(length)
    y = rng.uniform(size=400) < 1 / (1 + np.exp(-z))
    a, b, c = fit.fit_calibration(mean_post, length, y)
    assert b > 0.3 and c > 0.1


# -- E6: weak-label liveness model (off by default) ---------------------------------------------

def test_liveness_weak_labels_dead_wins():
    from possession import livemodel
    n = io.N_FRAMES
    first = io.halves()[0][1]
    b = {"speed": np.full(n, 9.0), "visible": np.ones(n, dtype=bool), "oob": np.zeros(n, dtype=bool)}
    speed = np.full(n, 3.0)
    b["oob"][first + 10:first + 20] = True
    gaps = [(first + 100, first + 200, 0.5, False), (first + 300, first + 400, 1.5, False)]
    y = livemodel.weak_labels(b, speed, gaps, kickoffs=[first + 1000])
    assert y[first] == 1 and y[first + 250] == 1               # visible, fast ball, players running
    assert np.all(y[first + 10:first + 20] == 0)               # out of pitch, however fast
    assert np.all(y[first + 100:first + 200] == 0)             # slow gap
    assert y[first + 350] == 1                                 # a gap at walking pace or more is not labelled dead
    assert np.all(y[first + 950:first + 1000] == 0) and y[first + 1000] == 1
    assert y[first - 1] == -1                                  # outside the graded blocks


def test_liveness_model_modes_and_default_off():
    from possession import livemodel, liveness, pipeline
    live = {"logodds_alive": np.array([1.0, 1.0, 1.0]), "pending": np.array([False, True, False])}
    b = {"oob": np.array([False, False, True])}
    p = np.array([0.9, 0.9, 0.9])
    laws = livemodel.combine(live, p, "replace+laws", b)["logodds_alive"]
    assert np.isclose(laws[0], np.log(9.0)) and laws[1] <= liveness.PENDING_LOGODDS
    assert laws[2] == liveness.OOB_LOGODDS
    assert np.allclose(livemodel.combine(live, p, "add", b)["logodds_alive"], 1.0 + np.log(9.0))
    assert pipeline.LIVENESS_MODEL is None                     # the submission uses the rule-based S2


def test_absorb_blips_only_sandwiched_team_spells():
    H, A, D = 0, 1, 2
    codes = np.array([H] * 10 + [A] * 3 + [H] * 10 + [A] * 3 + [D] * 5 + [A] * 3 + [H] * 4 + [A] * 6 + [H] * 2, dtype=np.int8)
    out = decoder.absorb_blips(codes, 3)
    assert (out[:23] == H).all()                       # H A H: the 3-frame A spell goes back to H
    assert (out[23:26] == A).all()                     # A next to a stoppage stays
    assert (out[26:31] == D).all() and (out[31:34] == A).all()
    assert (out[34:38] == H).all()                     # 4 frames: longer than the limit, stays
    assert (out[38:44] == A).all() and (out[44:] == H).all()   # block edge: no spell after it, stays
    assert (decoder.absorb_blips(codes, 2) == codes).all()     # nothing at most 2 frames long
