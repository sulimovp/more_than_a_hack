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

posteriors() runs forward-backward on the same chain (ROADMAP E5 step 1), and
remove_uncertain_spells() is the minimum-risk post-processing of E5 step 2. It is off
(MIN_RISK None) until the E1 transition labels set its threshold: analysis/e5_fit_min_risk.py.
"""

import math

import numpy as np

MIN_LIVE = 8
MIN_DEAD = 50
SWITCH = -6.0
LIVE_TO_DEAD = -15.0
DEAD_TO_LIVE = -15.0
FORCED = -1e4

# E5 step 2. A predicted possession whose P(real) is under MIN_RISK is removed; None is off.
# P(real) = s(a + b logit(mean posterior of its label) + c log(length in s)), (a, b, c) = P_REAL.
# The identity (0, 1, 0) is a placeholder: the threshold and (a, b, c) are fitted on the E1 labels only.
# MIN_RISK_MERGE: where a removed spell's frames go, "previous" (ROADMAP E5) or "posterior".
MIN_RISK = None
P_REAL = (0.0, 1.0, 0.0)
MIN_RISK_MERGE = "previous"
FPS = 25

# A team spell of at most BLIP_FRAMES with the other team on both sides (H A H or A H A, no stoppage) is
# given back to that team. Chosen on the hand-reviewed transitions (TECHNICAL.md, validation); None is off.
BLIP_FRAMES = None


def _emissions(alive, team, force_dead, normalise_team):
    """(e_h, e_a, e_d) log-emissions per frame, shared by viterbi() and posteriors()."""
    # P(H) = P(alive) P(H | alive): team evidence separates H from A but never makes a stoppage
    # look live, which is what a goalkeeper standing over a dead ball would otherwise do.
    e_live = -np.logaddexp(0.0, -alive)
    if normalise_team:
        e_h = e_live + np.minimum(0.0, team)
        e_a = e_live + np.minimum(0.0, -team)
    else:
        e_h = e_live - np.logaddexp(0.0, -team)
        e_a = e_live - np.logaddexp(0.0, team)
    e_d = -np.logaddexp(0.0, alive)
    e_h = np.where(force_dead, FORCED, e_h)
    e_a = np.where(force_dead, FORCED, e_a)
    return e_h, e_a, e_d


def viterbi(alive, team, enter_dead, leave_dead, restart_team, force_dead,
            min_live=MIN_LIVE, min_dead=MIN_DEAD, switch=SWITCH,
            live_to_dead=LIVE_TO_DEAD, dead_to_live=DEAD_TO_LIVE,
            switch_bonus=None, normalise_team=False):
    """Codes 0 (H), 1 (A), 2 (D) for every frame of one contiguous block.

    switch_bonus[t] is added to every transition cost into frame t (a change of label is more
    likely here). normalise_team keeps the team evidence from biasing live against dead: the
    favoured team's emission is log P(alive) exactly and the other team pays |team|. Without it
    H and A each carry log P(H | alive), which for neutral evidence is log 1/2 per frame
    against D, so a live stretch with no team evidence paid 0.69 nats per frame for being live.
    """
    n = len(alive)
    if switch_bonus is None:
        switch_bonus = np.zeros(n)
    k, m = min_live, min_dead
    H0, A0, D0 = 0, k, 2 * k
    Ht, At, Dt = k - 1, 2 * k - 1, 2 * k + m - 1
    n_states = 2 * k + m
    base = np.array([0] * k + [1] * k + [2] * m)
    e_h, e_a, e_d = _emissions(alive, team, force_dead, normalise_team)

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
        bonus = switch_bonus[t]
        into_live = dead_to_live + leave_dead[t] + bonus
        r = 0.5 * restart_team[t]
        cand = ((dp[At] + switch + bonus, At), (dp[Dt] + into_live + r, Dt))
        new[H0], bp[H0] = max(cand)
        cand = ((dp[Ht] + switch + bonus, Ht), (dp[Dt] + into_live - r, Dt))
        new[A0], bp[A0] = max(cand)
        into_dead = live_to_dead + enter_dead[t] + bonus
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


def _lse(*values):
    """log(sum(exp(values))) for a few scalars."""
    top = max(values)
    if top == -math.inf:
        return top
    return top + math.log(sum(math.exp(v - top) for v in values))


def posteriors(alive, team, enter_dead, leave_dead, restart_team, force_dead,
               min_live=MIN_LIVE, min_dead=MIN_DEAD, switch=SWITCH,
               live_to_dead=LIVE_TO_DEAD, dead_to_live=DEAD_TO_LIVE,
               switch_bonus=None, normalise_team=False):
    """(n, 3) array of P(H), P(A), P(D) for every frame of one block (ROADMAP E5 step 1).

    Forward-backward on the chain viterbi() decodes: the same states, emissions and transition
    scores, with log-sum-exp where viterbi() takes the max. The scores are not normalised
    transition probabilities, so these are the marginals of p(path) proportional to
    exp(score(path)), the distribution whose most likely path is the Viterbi path. Each frame's
    evidence is counted as if independent, so the posteriors are overconfident; E5 calibrates
    them on the E1 labels before any threshold is applied.
    """
    n = len(alive)
    k, m = min_live, min_dead
    if k < 2 or m < 2:
        raise ValueError("posteriors() needs chains of at least two states")
    bonus = np.zeros(n) if switch_bonus is None else np.asarray(switch_bonus, dtype=float)
    H0, A0, D0 = 0, k, 2 * k
    Ht, At, Dt = k - 1, 2 * k - 1, 2 * k + m - 1
    n_states = 2 * k + m
    chains = ((H0, Ht), (A0, At), (D0, Dt))
    e_h, e_a, e_d = _emissions(alive, team, force_dead, normalise_team)
    emit = np.empty((n, n_states))
    emit[:, :k] = e_h[:, None]
    emit[:, k:2 * k] = e_a[:, None]
    emit[:, 2 * k:] = e_d[:, None]
    sw = (switch + bonus).tolist()
    into_live = (dead_to_live + np.asarray(leave_dead, dtype=float) + bonus).tolist()
    r = (0.5 * np.asarray(restart_team, dtype=float)).tolist()
    into_dead = (live_to_dead + np.asarray(enter_dead, dtype=float) + bonus).tolist()

    # forward, rescaled every frame (the per-frame normalisation below absorbs the constants)
    fwd = np.empty((n, n_states))
    a = np.full(n_states, -np.inf)
    a[[Ht, At, Dt]] = 0.0
    for t in range(n):
        new = np.empty(n_states)
        for first, tail in chains:
            new[first + 1:tail] = a[first:tail - 1]
            new[tail] = np.logaddexp(a[tail - 1], a[tail])
        new[H0] = _lse(a[At] + sw[t], a[Dt] + into_live[t] + r[t])
        new[A0] = _lse(a[Ht] + sw[t], a[Dt] + into_live[t] - r[t])
        new[D0] = _lse(a[Ht] + into_dead[t], a[At] + into_dead[t])
        a = new + emit[t]
        a -= a.max()
        fwd[t] = a

    # backward, combined with the forward pass frame by frame
    post = np.empty((n, 3))
    b = np.zeros(n_states)
    for t in range(n - 1, -1, -1):
        g = fwd[t] + b
        g = np.exp(g - g.max())
        post[t] = (g[:k].sum(), g[k:2 * k].sum(), g[2 * k:].sum())
        if t == 0:
            break
        h = emit[t] + b
        new = np.empty(n_states)
        for first, tail in chains:
            new[first:tail] = h[first + 1:tail + 1]
        new[Ht] = _lse(h[Ht], sw[t] + h[A0], into_dead[t] + h[D0])
        new[At] = _lse(h[At], sw[t] + h[H0], into_dead[t] + h[D0])
        new[Dt] = _lse(h[Dt], into_live[t] + r[t] + h[H0], into_live[t] - r[t] + h[A0])
        b = new - new.max()
    return post / post.sum(axis=1, keepdims=True)


def runs(codes):
    """[(start, stop, code)] maximal constant runs of one block."""
    codes = np.asarray(codes)
    if not len(codes):
        return []
    bounds = np.concatenate(([0], np.flatnonzero(codes[1:] != codes[:-1]) + 1, [len(codes)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(codes[bounds[i]])) for i in range(len(bounds) - 1)]


def p_real(mean_post, length_s, coef=None):
    """Calibrated P(a predicted possession is real) from its mean posterior and length (E5)."""
    a, b, c = P_REAL if coef is None else coef
    p = np.clip(mean_post, 1e-6, 1.0 - 1e-6)
    z = a + b * np.log(p / (1.0 - p)) + c * np.log(np.maximum(length_s, 1.0 / FPS))
    return 1.0 / (1.0 + np.exp(-z))


def _p_real_scalar(mean_post, length_s, coef):
    a, b, c = coef
    p = min(max(mean_post, 1e-6), 1.0 - 1e-6)
    z = a + b * math.log(p / (1.0 - p)) + c * math.log(max(length_s, 1.0 / FPS))
    return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


def spell_posteriors(codes, post):
    """[(start, stop, code, mean posterior of code, min posterior of code)] for every run of one block."""
    return [(a, z, c, float(post[a:z, c].mean()), float(post[a:z, c].min())) for a, z, c in runs(codes)]


def remove_uncertain_spells(codes, post, threshold, coef=None, merge=None,
                            min_live=MIN_LIVE, min_dead=MIN_DEAD):
    """E5 step 2, minimum-risk post-processing of one block's codes. threshold None is a no-op.

    A wrong short spell costs spellF1 a false positive and splits the true possession around it,
    so the possession (H or A run) with the lowest P(real) is removed while that is under
    threshold, one at a time, P(real) recomputed after each merge. With merge "previous" (the
    roadmap's rule) its frames go to the previous team, to the next team when the previous label
    is D or the block starts there, and to D between two stoppages; with "posterior" they go to
    whichever neighbouring label has the larger posterior mass over them. Merging only lengthens
    runs, so the minimum durations still hold; the check at the end makes sure.
    """
    if threshold is None:
        return codes
    merge = MIN_RISK_MERGE if merge is None else merge
    if merge not in ("previous", "posterior"):
        raise ValueError("merge must be 'previous' or 'posterior'")
    coef = P_REAL if coef is None else coef
    mass = np.vstack([np.zeros((1, 3)), np.cumsum(post, axis=0)])  # posterior mass of [a, z) by prefix sums
    segs = [list(s) for s in runs(codes)]

    def prob(seg):
        a, z, c = seg
        if c == 2:
            return math.inf
        return _p_real_scalar((mass[z, c] - mass[a, c]) / (z - a), (z - a) / float(FPS), coef)

    probs = [prob(s) for s in segs]
    while segs:
        i = min(range(len(segs)), key=probs.__getitem__)  # first of the lowest, as a left-to-right scan
        if probs[i] >= threshold:
            break
        a, z, _ = segs[i]
        prev = segs[i - 1][2] if i > 0 else None
        nxt = segs[i + 1][2] if i + 1 < len(segs) else None
        if merge == "posterior":
            sides = [c for c in (prev, nxt) if c is not None] or [2]
            fill = max(sides, key=lambda c: (mass[z, c] - mass[a, c], c == prev))
        else:
            fill = prev if prev in (0, 1) else (nxt if nxt in (0, 1) else 2)
        segs[i][2] = fill
        if i + 1 < len(segs) and segs[i + 1][2] == fill:
            segs[i][1] = segs[i + 1][1]
            del segs[i + 1], probs[i + 1]
        if i > 0 and segs[i - 1][2] == fill:
            segs[i - 1][1] = segs[i][1]
            del segs[i], probs[i]
            i -= 1
        probs[i] = prob(segs[i])
    out = np.empty(len(codes), dtype=np.int8)
    for a, z, c in segs:
        out[a:z] = c
    for a, z, c in segs[1:-1]:
        if z - a < (min_dead if c == 2 else min_live):
            raise AssertionError("run {}-{} of {} is under its minimum duration".format(a, z, c))
    return out


def absorb_blips(codes, max_frames):
    """Codes with every team spell of at most max_frames that sits between two spells of the other team
    relabelled as that team. One pass over the original runs; dead spells and block edges are untouched."""
    out = np.array(codes, copy=True)
    r = runs(codes)
    for k in range(1, len(r) - 1):
        start, stop, code = r[k]
        before, after = r[k - 1][2], r[k + 1][2]
        if code in (0, 1) and before == after and before in (0, 1) and before != code and stop - start <= max_frames:
            out[start:stop] = before
    return out
