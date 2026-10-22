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
from possession import decoder, io  # noqa: E402

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
