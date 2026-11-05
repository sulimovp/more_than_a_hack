# Ball possession from automatic tracking

Method note for More than a Hack 2027, phase 1. The ranked file is one label per tactical
frame. This note is the description that goes with it.

## Approach

Each half is decoded offline as a three-state sequence, home, away or dead. A possession is a
maximal run of one team with the ball in play, so the decoder is built to produce those runs
rather than a label per frame that is then smoothed.

Four per-frame signals are combined by Viterbi:

- **Liveness.** Log-odds that the ball is in play. Inside a long tracking gap the decision is the
  players' median speed, because on this match the tracker withholds the ball for most of a
  stoppage and the players walk. With the ball visible, the same speed is a weak signal, and a
  moving ball adds a capped bonus so that a pass is not called dead when the players have slowed
  down. The bonus is small enough that a one-second kick inside a real stoppage does not split it.
  Out-of-pitch frames, and the stretch from the ball leaving the pitch until it is struck back
  into play, are dead. Whistles from the tactical audio (narrow band, 2.5–4.5 kHz) add a bonus on
  the live-to-dead transition. They do not decide a frame by themselves.
- **Control.** Who last had the ball, as log-odds of home against away. A player inside 1.5 m
  contributes exp(−d / 0.7 m). If both teams are in that zone the score shrinks to a nudge
  toward the nearer one. A touch — a peak in ball acceleration above 20 m/s², a clear change of
  direction, or the ball staying at a player's feet — sets that team, and the assignment is held
  until the next touch. That hold is what carries a pass. Relative speed above 8 m/s is treated as
  a deflection and is not a touch. While the ball is faster than 8 m/s and nobody is within 2 m,
  the zone contributes nothing.
- **Restarts.** At a dead-to-live transition the restarting team is taken from the laws and from
  Smart Tagging: the first-half kickoff, the opposite team after the break and after a goal, the
  tagged team for a corner, goal kick or free kick, and otherwise the player standing over the
  still ball in the seconds before it is struck, or failing that the player nearest the ball when
  it is struck. Every throw-in is tagged without a team, so those fall through to the taker.
- **Team shape.** Where the ball is missing during a live phase, a gradient-boosted model gives a
  weak team score from the two teams' positions and velocities, oriented by attack direction. It is
  trained on frames where a player is at the ball, each half predicted by a model fit on the other
  half, and its probability is rounded to 0.02 so that a last-bit difference between machines
  cannot change the file. The weight is small (a confident five-second gap is not enough to pay for two switches), so
  shape only settles a handover the geometry left open.

The chain forces a live spell to last at least 8 frames (0.3 s) and a stoppage at least 50 frames
(2 s). Because the decode sees the whole half, a switch is dated to the first frame of evidence.
Frames outside the two graded blocks are left as no answer. Frames 110808–113741, the graded
portion of the second half before its kickoff, are dead.

## Assets

Used: the tactical tracking, its team metadata, the tactical homography only inside the labelling
tool, Smart Tagging (set pieces, goals, and phase tags as a weak check), and the audio track of
the tactical video.

Not used: the broadcast video and its tracking. They are on another timeline, with cuts and
replays, and a third of the broadcast frames have no homography. The zone codes in Smart Tagging
(final third, players in the box) are not used as team evidence; on the public window they agree
with the true team about half the time. Defending-phase tags name the defending team and are
inverted when they are used as a check.

No hand label enters the file. The public five minutes are a set of cases, not a training set.

## Possession rules

The ball stays with the last team through a loose ball, a flight, and a deflection. A change
requires a touch or a player staying with the ball, not a single nearer frame. A duel in which
both teams are within 1.5 m does not by itself change the team. A stoppage is a positive event:
the ball outside the pitch, a restart that has not been taken, a long gap in which the players are
walking, or the whistle together with that context. The same team restarting after a stoppage is
two possessions. Dead-ball ownership is not predicted.

## What the public window shows

On the 17 published possessions the file scores spellF1 0.506 and macroF1 0.888. The published
nearest-player floor scores 0.326 and 0.621 on the same window. All four stoppages match, at
temporal IoU 0.81 to 0.94. The count of possessions matches, 17 against 17.

Five possessions do not match. In each case the automatic tracking does not put the true team at
the ball. Two of them are under 1.5 s and the nearest player of the true team is never within
control distance, or is only there while the ball is moving past at 7–10 m/s. One 8 s spell is a
flight in which the other team is nearer until a touch 4 s after the ground-truth change. These
are disagreements between the automatic ball track and the corrected track the ground truth was
derived from. Holding the last team, which is the right rule for a loose ball, keeps the previous
team when the track never shows the new one.

Over the full match the file has 313 possessions, median length 7.0 s, the ball dead for 49.5% of
graded time, a restart after 57 of 62 set pieces and goals, and 81.5% agreement with the Smart
Tagging phase tags on live frames. None of these is the score. The full-match ground truth is
private. A spellF1 gap below about 0.04 on a full match is noise, and the public window is
noisier than that: its bootstrap band on the previous file was roughly 0.35–0.67.

## Limitations

The control signal cannot recover a possession the tracking does not show. Short duels and blocked
passes, where the corrected track and the automatic track disagree about who is at the ball, are
the misses that remain. A ball detector or the broadcast camera would only help where the ball is
missing; the misses above are on frames where the ball is already tracked.

The whistle score does not change the decode. Stoppage length is set by the tracking gaps and by
player speed.

Liveness is rule-based. A gradient-boosted liveness model was built and is not used. Its training labels
were rules: dead when the ball is out of the pitch, during slow tracking gaps and just before a kickoff;
live when a fast ball is among running players. The model can tell those labels apart from the very
features that define them, so it repeats the rules everywhere else and calls 57-60% of graded time dead.
For the same reason the decoder keeps its unnormalised team emission, in which a live frame with no team
evidence pays log 2 against dead. Normalising it moved two of the four public stoppage onsets later and
added possessions; under the learned liveness, whose log-odds dwarf log 2, it changed almost nothing. A
learned liveness needs labelled frames where those rules say nothing.

There is no hand-labelled sample beyond the public five minutes, so the transition costs and the
speed thresholds have not been calibrated. They are physical values, and the sensitivity of the
earlier file is in `analysis/sensitivity.md`. Moving them to chase the public window is how the
published baselines pick the wrong smoothing width.

## Reproduction

```bash
pip install -r requirements-pipeline.txt
python -m possession.make_submission --team <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

The match bundle belongs in `data/` as described in `DATA.md`. The command rewrites the CSV from
that bundle. CPU only, deterministic. `python -m pytest tests -q` checks the decoder and that a
fresh run matches `tests/expected_sha256.txt`. Caches under `data/cache/` are derived match data
and are not part of the repository.
