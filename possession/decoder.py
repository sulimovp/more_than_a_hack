"""S6: offline Viterbi over H / A / D with minimum durations and time-varying transitions.

Each base state is a count-up chain (H_1..H_k, A_1..A_k, D_1..D_m) whose last state loops on itself,
so a spell cannot be shorter than its chain. The chain lengths are the only hard constraints:
k = 8 frames keeps 0.3 s duel spells legal (the public window has a 0.76 s one), m = 50 frames
says no stoppage is shorter than 2 s.

The decoder sees the whole half at once, so a handover lands on the first frame of evidence for
it rather than after a confirmation window: one second of lag would cost a third of spellF1.

Emissions per frame, with s the logistic function (alive and team are log-odds):
    H: log s(alive) + log s(team)    A: log s(alive) + log s(-team)    D: log s(-alive)
Transitions into frame t:
    H_k <-> A_1                 SWITCH
    H_k, A_k -> D_1             LIVE_TO_DEAD + enter_dead[t]
    D_m -> H_1 / A_1            DEAD_TO_LIVE + leave_dead[t] +/- restart_team[t]/2
"""

import numpy as np

MIN_LIVE = 8
MIN_DEAD = 50
SWITCH = -6.0
LIVE_TO_DEAD = -15.0
DEAD_TO_LIVE = -15.0
FORCED = -1e4


def viterbi(alive, team, enter_dead, leave_dead, restart_team, force_dead,
            min_live=MIN_LIVE, min_dead=MIN_DEAD, switch=SWITCH,
            live_to_dead=LIVE_TO_DEAD, dead_to_live=DEAD_TO_LIVE):
    """Codes 0 (H), 1 (A), 2 (D) for every frame of one contiguous block."""
    n = len(alive)
    k, m = min_live, min_dead
    H0, A0, D0 = 0, k, 2 * k
    Ht, At, Dt = k - 1, 2 * k - 1, 2 * k + m - 1
    n_states = 2 * k + m
    base = np.array([0] * k + [1] * k + [2] * m)

    # P(H) = P(alive) P(H | alive): team evidence separates H from A but never makes a stoppage
    # look live, which is what a goalkeeper standing over a dead ball would otherwise do.
    e_live = -np.logaddexp(0.0, -alive)
    e_h = e_live - np.logaddexp(0.0, -team)
    e_a = e_live - np.logaddexp(0.0, team)
    e_d = -np.logaddexp(0.0, alive)
    e_h = np.where(force_dead, FORCED, e_h)
    e_a = np.where(force_dead, FORCED, e_a)

    neg = -np.inf
    dp = np.full(n_states, neg)
    dp[[Ht, At, Dt]] = 0.0  # a block may open mid-spell, so start on a chain's tail
    back = np.zeros((n, n_states), dtype=np.int16)
    chain_from = np.arange(-1, n_states - 1)

    for t in range(n):
        new = np.full(n_states, neg)
        bp = np.zeros(n_states, dtype=np.int16)
        # walk along each chain
        for first, tail in ((H0, Ht), (A0, At), (D0, Dt)):
            new[first + 1:tail + 1] = dp[first:tail]
            bp[first + 1:tail + 1] = chain_from[first + 1:tail + 1]
            if dp[tail] > new[tail]:
                new[tail] = dp[tail]
                bp[tail] = tail
        # entries
        into_live = dead_to_live + leave_dead[t]
        r = 0.5 * restart_team[t]
        cand = ((dp[At] + switch, At), (dp[Dt] + into_live + r, Dt))
        new[H0], bp[H0] = max(cand)
        cand = ((dp[Ht] + switch, Ht), (dp[Dt] + into_live - r, Dt))
        new[A0], bp[A0] = max(cand)
        into_dead = live_to_dead + enter_dead[t]
        cand = ((dp[Ht] + into_dead, Ht), (dp[At] + into_dead, At))
        new[D0], bp[D0] = max(cand)

        emit = np.where(base == 0, e_h[t], np.where(base == 1, e_a[t], e_d[t]))
        dp = new + emit
        back[t] = bp

    state = int(np.argmax(dp))
    path = np.empty(n, dtype=np.int16)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return base[path].astype(np.int8)
