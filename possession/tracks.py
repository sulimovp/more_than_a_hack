"""Player-track repairs applied before any stage reads the tracking (ROADMAP.md E4b).

The goalkeeper slots sometimes sit on the other team's keeper: the away keeper slot is in the home
team's box for the first 59 s of the public window, and the home keeper slot is on Athletic's keeper
at 15604-15940 (analysis/gt_convention.md). A keeper never stands in the opponent's penalty box in
open play, so on those frames the slot is dropped (NaN) rather than trusted for the wrong team.
"""

import numpy as np

from . import io

BOX_DEPTH = 18.0  # metres from the goal line, the penalty box


def _in_box(x, left):
    return x < BOX_DEPTH if left else x > io.PITCH[0] - BOX_DEPTH


def keeper_slots(players, team, first, stop, home_defends_left):
    """{team code: slot} of the slot of each team that stands in its own box most often in this half."""
    out = {}
    for code, left in ((io.HOME, home_defends_left), (io.AWAY, not home_defends_left)):
        best, best_share = None, 0.0
        for k in np.flatnonzero(team == code):
            x = players[first:stop, k, 0]
            x = x[~np.isnan(x)]
            if len(x) < 25 * io.FPS:
                continue
            share = np.mean(_in_box(x, left))
            if share > best_share:
                best, best_share = k, share
        if best is not None and best_share > 0.5:
            out[code] = best
    return out


def repair_keepers(players, team):
    """A copy of players with each keeper slot set to NaN while it stands in the opponent's box."""
    players = players.copy()
    for half, first, stop in io.halves():
        home_left = io.home_attacks_right(half)
        for code, k in keeper_slots(players, team, first, stop, home_left).items():
            own_left = home_left if code == io.HOME else not home_left
            x = players[first:stop, k, 0]
            wrong = ~np.isnan(x) & _in_box(x, not own_left)
            players[first:stop, k][wrong] = np.nan
    return players
