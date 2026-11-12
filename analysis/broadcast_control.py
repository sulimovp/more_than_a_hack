"""The broadcast view in the control stage: the veto that is kept, and broadcast stand-ins that were tested and not kept.

    python -m analysis.broadcast_control

Veto (control.contradicted): how many confirmed touches it drops, the file without it and with VETO_PLAYER_M 2.5 m,
and on the frames it changes whether the Smart Tagging phase tags side with the new label or the old one.

Stand-ins: where the two balls are within 3 m and the broadcast has a player of team T within 1.5 m of its ball, while
the tactical view has no player of T within 3 m of that spot, the tactical view is taken to be missing that player.
The broadcast player is added as an extra tactical player, at the tactical ball plus its offset from the broadcast
ball, and the control evidence (zone, touches, confirmation, veto) is computed on the augmented players.

Each variant prints the public window and the full-match weak checks next to the submission. Aggregates only.
"""

import numpy as np

from possession import ball, control, io, make_submission, pipeline, report

import possession_metrics as pm  # noqa: E402  (io put evaluation/ on the path)

AT_BALL_M = control.VETO_PLAYER_M
MISSING_M = 3.0
PUBLIC_CASES = ((11351, 11466), (15766, 15799))


def augmented(b, players, team, second):
    """(players, team, stand-ins): the tactical players and teams with the broadcast slots that ever stand in
    appended, and those slots alone. A slot is NaN where it does not stand in."""
    bb, bp = second["ball"], second["players"]
    with np.errstate(invalid="ignore"):
        same = np.linalg.norm(b["ball"] - bb, axis=1) < control.VETO_BALL_M
    offset = bp - bb[:, None, :]
    at_ball = same[:, None] & (np.nan_to_num(np.linalg.norm(offset, axis=2), nan=np.inf) < AT_BALL_M)
    pos = b["ball"][:, None, :] + offset
    out = np.full(bp.shape, np.nan, dtype=np.float32)
    for k in range(out.shape[1]):
        rows = np.flatnonzero(at_ball[:, k])
        mates = players[rows][:, team == second["team"][k]]
        d = np.linalg.norm(mates - pos[rows, k][:, None, :], axis=2)
        missing = ~(np.nan_to_num(d, nan=np.inf) < MISSING_M).any(axis=1)
        out[rows[missing], k] = pos[rows[missing], k]
    used = ~np.all(np.isnan(out[:, :, 0]), axis=0)
    return np.concatenate([players, out[:, used]], axis=1), np.concatenate([team, second["team"][used]]), out[:, used]


def how_often(b, players, team, second, touches):
    """Frames where the broadcast's nearest player is at its ball and the tactical view has none of that team near."""
    graded = io.graded_mask()
    with np.errstate(invalid="ignore"):
        same = np.linalg.norm(b["ball"] - second["ball"], axis=1) < control.VETO_BALL_M
    near = {}
    for name, ball_xy, p, t in (("bc", second["ball"], second["players"], second["team"]),
                                ("tac", b["ball"], players, team)):
        d = np.nan_to_num(np.linalg.norm(p - ball_xy[:, None, :], axis=2), nan=np.inf)
        near[name] = [np.where(t == side, d, np.inf).min(axis=1) for side in (io.HOME, io.AWAY)]
    side = np.where(near["bc"][0] < near["bc"][1], io.HOME, io.AWAY)
    at_ball = graded & same & (np.minimum(*near["bc"]) < AT_BALL_M)
    tac_same_team = np.where(side == io.HOME, near["tac"][0], near["tac"][1])
    print("graded frames, balls within {:.0f} m and a broadcast player within {:.1f} m: {}".format(
        control.VETO_BALL_M, AT_BALL_M, int(at_ball.sum())))
    for radius in (2.0, MISSING_M, 5.0):
        sit = at_ball & ~(tac_same_team < radius)
        print("  no tactical player of that team within {:.0f} m: {} ({:.1%}; home {}, away {}); at confirmed touches {}".format(
            radius, int(sit.sum()), sit.sum() / at_ball.sum(), int((sit & (side == io.HOME)).sum()),
            int((sit & (side == io.AWAY)).sum()), int(sit[touches].sum())))
        for a, z in PUBLIC_CASES:
            print("    {}-{}: {} of {} frames".format(a, z, int(sit[a:z].sum()), z - a))


def phase_expected(tags):
    """Team the possession-phase tags name on each frame; -1 untagged, -2 where tags conflict."""
    expected = np.full(io.N_FRAMES, -1, dtype=np.int8)
    for t in tags:
        sign = report.POSSESSION_PHASES.get(t["code"])
        team = report.TEAM_CODE.get(t["tags"].get("Team"))
        if sign is None or team is None:
            continue
        e = team if sign == 1 else 1 - team
        window = expected[t["start_frame"]:t["end_frame"]]
        window[:] = np.where((window == -1) | (window == e), e, -2)
    return expected


def phase_shift(new, old, tags):
    """On graded frames that switch team between old and new: which one the phase tags side with."""
    expected = phase_expected(tags)
    flip = io.graded_mask() & (new != old) & (new < io.DEAD) & (old < io.DEAD)
    tagged = flip & (expected >= 0)
    towards = away = 0
    for a, z in ball.runs(flip):
        m = expected[a:z] >= 0
        if m.sum() >= 0.5 * (z - a):
            if np.mean(new[a:z][m] == expected[a:z][m]) > 0.5:
                towards += 1
            else:
                away += 1
    print("team switches {}; inside an unambiguous phase tag {}: the tag sides with the new label on {}, the old on {};"
          " stretches mostly inside tags: {} towards the tags, {} away".format(
              int(flip.sum()), int(tagged.sum()), int(np.sum(new[tagged] == expected[tagged])),
              int(np.sum(old[tagged] == expected[tagged])), towards, away))


def decode(f, ctrl):
    f = dict(f, control=ctrl)
    return pipeline.decode(f, pipeline.add_shape(f, pipeline.decode(f)))


def show(name, codes, ref, tags):
    res = pm.evaluate(io.load_gt_public(), codes, bootstrap_n=0)
    graded = io.graded_mask()
    print("== {}: public spellF1 {:.4f} macroF1 {:.4f}; graded frames changed {}, public frames changed {}; sha256 {}".format(
        name, res["spell_f1_mean"], res["macro_f1"], int(np.sum(codes[graded] != ref[graded])),
        int(np.sum(codes[11185:18685] != ref[11185:18685])), make_submission.codes_sha256(codes)[:12]))
    report.weak_checks(codes, tags)
    report.public_unit_tests(codes)


def main():
    f = pipeline.features()
    tags = io.load_tags()
    b, players, team, second = f["ball"], f["players"], f["team"], f["broadcast"]
    dist, rel = control._closest(b, players, team)
    before = control.touch_events(b, dist, rel, players, team)
    submission = decode(f, f["control"])
    show("submission", submission, submission, tags)

    print("confirmed touches {}, vetoed {}".format(len(before[0]), int(control.contradicted(b, *before, second).sum())))
    no_veto = decode(f, control.evidence(b, players, team))
    show("without the veto", no_veto, submission, tags)
    phase_shift(submission, no_veto, tags)
    kept = control.VETO_PLAYER_M
    control.VETO_PLAYER_M = 2.5
    print("VETO_PLAYER_M 2.5: vetoed {}".format(int(control.contradicted(b, *before, second).sum())))
    wide = decode(f, control.evidence(b, players, team, second))
    control.VETO_PLAYER_M = kept
    show("veto within 2.5 m", wide, submission, tags)

    how_often(b, players, team, second, before[0])
    aug_players, aug_team, extra = augmented(b, players, team, second)
    graded = io.graded_mask()
    used = ~np.isnan(extra[:, :, 0])
    print("graded frames with a stand-in: {} ({:.1f} min), broadcast slots used {}".format(
        int(used[graded].any(axis=1).sum()), used[graded].any(axis=1).sum() / 1500.0, extra.shape[1]))
    dist, rel = control._closest(b, aug_players, aug_team)
    after = control.touch_events(b, dist, rel, aug_players, aug_team)
    s0, s1 = set(zip(*[x.tolist() for x in before])), set(zip(*[x.tolist() for x in after]))
    print("confirmed touches {} -> {}: {} gone, {} new".format(len(s0), len(s1), len(s0 - s1), len(s1 - s0)))
    for a, z in PUBLIC_CASES:
        print("  {}-{}: stand-in frames {}; touches within 2 s before {}, after {}".format(
            a, z, int(used[a:z].any(axis=1).sum()), sorted(t for t in s0 if a - 50 <= t[0] < z),
            sorted(t for t in s1 if a - 50 <= t[0] < z)))
    show("with stand-ins", decode(f, control.evidence(b, aug_players, aug_team, second)), submission, tags)


if __name__ == "__main__":
    main()
