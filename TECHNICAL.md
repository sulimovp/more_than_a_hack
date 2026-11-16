# Ball possession from automatic tracking

Team Arraxis: Pavel Sulimov, Sergey Eremeykin, Daniil Sulimov.  
Code: <https://github.com/sulimovp/more_than_a_hack>

Method note for More than a Hack 2027, phase 1. The ranked file is one label per tactical
frame. This note is the description that goes with it.

## Approach

Each half is decoded offline as a three-state sequence, home, away or dead. A possession is a
maximal run of one team with the ball in play, so the decoder is built to produce those runs
rather than a label per frame that is then smoothed.

Before any signal is computed, the goalkeeper tracks are repaired. On this match the slot of
one team's keeper sometimes follows the other team's keeper for a minute or more, which hands
that team every ball the opposing keeper touches. In each half the slot of each team that
stands most often in its own penalty box is taken as its keeper, and on frames where that slot
is inside the opponent's penalty box its position is dropped. A keeper does not stand in the
opponent's box in open play, so a position there is a tracking error.

Four per-frame signals are combined by Viterbi:

- **Liveness.** Log-odds that the ball is in play. Inside a long tracking gap the decision is the
  players' median speed, because on this match the tracker withholds the ball for most of a
  stoppage and the players walk. With the ball visible, the same speed is a weak signal, and a
  moving ball adds a capped bonus so that a pass is not called dead when the players have slowed
  down. The bonus is small enough that a one-second kick inside a real stoppage does not split it.
  Out-of-pitch frames, and the stretch from the ball leaving the pitch until it is struck back
  into play, are dead. Whistles from the tactical audio (narrow band, 2.5 to 4.5 kHz) add a bonus
  on the live-to-dead transition. They do not decide a frame by themselves.
- **Control.** Who last had the ball, as log-odds of home against away. A player inside 1.5 m
  contributes exp(−d / 0.7 m). If both teams are in that zone the score shrinks to a nudge
  toward the nearer one. A touch sets that team, and the assignment is held until the next
  touch; that hold is what carries a pass. A touch is a peak in ball acceleration above
  20 m/s², a clear change of direction, or the ball staying at a player's feet, with a player
  within 1.5 m. Relative speed above 8 m/s is treated as a deflection and is not a touch. While
  the ball is faster than 8 m/s and nobody is within 2 m, the zone contributes nothing.
- **Restarts.** At a dead-to-live transition the restarting team is taken from the laws and from
  Smart Tagging: the first-half kickoff, the opposite team after the break and after a goal, the
  tagged team for a corner, goal kick or free kick, and otherwise the player standing over the
  still ball in the seconds before it is struck, or failing that the player nearest the ball when
  it is struck. Every throw-in is tagged without a team, so those fall through to the taker. The
  first-half kickoff team, Athletic, is a fixed fact about this match read from the video; it is
  not inferred from the data.
- **Team shape.** Where the ball is missing during a live phase, a gradient-boosted model gives a
  weak team score from the two teams' positions and velocities, oriented by attack direction. It is
  trained on frames where a player is at the ball, each half predicted by a model fit on the other
  half, and its probability is rounded to 0.02 so that a last-bit difference between machines
  cannot change the file. The weight is small (a confident five-second gap is not enough to pay
  for two switches), so shape only settles a handover the geometry left open.

The chain forces a live spell to last at least 8 frames (0.3 s) and a stoppage at least 50 frames
(2 s). Because the decode sees the whole half, a switch is dated to the first frame of evidence.
Frames outside the two graded blocks are left as no answer. Frames 110808 to 113741, the graded
portion of the second half before its kickoff, are dead.

After the decode, a team spell shorter than 3 s that has the other team on both sides, with no
stoppage in between, is given back to that team. It is the possession-level form of the 2 s
first-controlled-action rule for touches: a brief spell inside the other team's possession is almost
always a block or an uncontrolled touch. The threshold was chosen on the hand-reviewed transitions
(validation section). On this match it changes 10 spells and 554 graded frames and no public frame.

## Possession rules

The ball stays with the last team through a loose ball, a flight, and a deflection. A change
requires a touch or a player staying with the ball, not a single nearer frame. The ground truth
moves possession on the new team's first controlled action, so a touch counts only if the other
team does not touch the ball within the next 2 s; a block, a deflection or a lost challenge is
dropped before it reaches the hold. One exception concerns the ball reappearing: when it is
seen again after at least three missing frames with a player within 0.5 m, that frame is a
touch by that player whatever the relative speed, because the ball is often first detected as it
leaves the kicker's foot.

A duel in which both teams are within 1.5 m does not by itself change the team. A stoppage is a
positive event: the ball outside the pitch, a restart that has not been taken, a long gap in which
the players are walking, or the whistle together with that context. The same team restarting after
a stoppage is two possessions. Dead-ball ownership is not predicted.

## The broadcast tracking as a second view

The broadcast recording comes with its own automatic tracking in the same format and the same
pitch coordinates as the tactical one, but on its own frame numbers. Within a half the two
recordings run at the same rate, so a single offset per half puts the broadcast on the tactical
timeline. The offset is not given with the data. It is found as the shift that puts the two ball
tracks within 3 m of each other on the largest share of frames, searched over every shift that
overlaps the half, with ties going to the smaller shift. On this match the tactical frame is the
broadcast frame plus 4014 in the first half and plus 26920 in the second. At those offsets 80% and
78% of the frames on which both balls are visible agree within 3 m, and the median distance between
the two balls is 0.8 m. The best offset in every 3-minute window except one lies within 3 frames of
the half's offset. The best share more than 50 frames away is under 9%, so the alignment is not
ambiguous. Replays and cut-aways lower the share but do not move the offset.
Team slots follow the same convention as the tactical metadata, and the keeper repair above is
applied to the broadcast players too.

The broadcast is used in one place: it can veto a touch. A touch is dropped when, at its frame,
the broadcast ball is within 3 m of the tactical ball, so both views see the same ball, and the
broadcast's nearest player is within 1.5 m of it and belongs to the other team. The broadcast
never adds a touch or names a team on its own. On this match the veto removes 26 of 1,044
touches. No frame of the public window changes; over the full match 1,071 graded frames change,
the possession count falls from 274 to 259, and agreement with the Smart Tagging phase tags rises
from 81.8% to 82.6%. The changed frames inside the phase tags fall in nine stretches, six of which
move towards the tags and three away from them, so this is modest evidence.

On the public window the broadcast ball is visible on 90.5% of live frames and 5.7% of dead ones.
Where both views have a player within 1 m of their ball and name the same team, the true team
matches on 98.1% of frames, against 92.8% for the tactical view alone. Where they name different
teams, the tactical view is right 46% of the time and the broadcast 54%. Neither view can be
trusted on such a frame, which is why a disagreement only removes a touch and does not hand it to
the other team.

Several stronger uses were tried and dropped. A liveness score learned from the broadcast alone
separated live from dead play well, but adding it to the liveness moved public stoppage edges:
replays run into restarts and the wide shot often returns before a free kick is taken, so a
stoppage ended seconds early or late. Computing the control signal on the broadcast and averaging
or adding it to the tactical one lost two public possessions. The last attempt added a broadcast
player as an extra tactical player wherever the tactical view had no player of that team within
3 m of the ball. It moved one public transition 65 frames closer to the truth, but it did not
recover the possession behind it, and phase-tag agreement fell from 82.6% to 82.4%.

## Assets

Used: the tactical tracking and its team metadata; the broadcast tracking and its team metadata,
for the veto above; Smart Tagging (set pieces, goals, and phase tags as a weak check); and the
audio track of the tactical video. The tactical homography is used only inside the labelling tool.
The broadcast video, its audio and its homography are not used. The zone codes in Smart Tagging
(final third, players in the box) are not used as team evidence; on the public window they agree
with the true team about half the time. Defending-phase tags name the defending team and are
inverted when they are used as a check.

No hand label enters the file as a label. One threshold, the length of the short spells given back
to the surrounding team, was chosen on hand-reviewed transitions, which the rules allow for
validation. The public five minutes are a set of cases, not a training set.

## What the public window shows

On the public window the file scores spellF1 0.607 and macroF1 0.930. The published
nearest-player floor scores 0.326 and 0.621 on the same window. All four stoppages match, at
temporal IoU 0.81 to 0.94.

The file has fewer possessions than the truth in the window, 12 against 17. Nine of the 17 true
possessions are matched at IoU 0.5 or more and three more at 0.38 to 0.47; five are missed. The
shortfall comes mostly from the first-controlled-action rule, which absorbs a spell that the other
team ends within 2 s. In the ground truth some of those short spells are possessions: a tackle that
is credited until the next kick, or a single poke that wins the ball before the opponent recovers
it. Three of the missed possessions last 0.8 s, 1.3 s and 2.8 s. The 3.6 s miss is the away spell
that follows the 1.3 s home one; the file keeps the away team across both, as part of one longer
spell, so neither is matched. In the 8.1 s
miss the automatic tracking puts a player of the other team nearest the ball on every frame where
anyone is within 2.5 m of it. Holding the last team, which is the right rule for a
loose ball, keeps the previous team when the tracking never shows the new one at the ball.

Over the full match the file has 239 possessions, median length 9.3 s, the ball dead for 49.4% of
graded time, a restart after 58 of 62 set pieces and goals, and 83.4% agreement with the Smart
Tagging phase tags on live frames. None of these is the score. The full-match ground truth is
private. A spellF1 gap below about 0.04 on a full match is noise, and the public window is
noisier than that: its bootstrap band on an earlier file was roughly 0.35 to 0.67.

## Validation on hand-reviewed transitions

Outside the public five minutes there is no ground truth, so one of us reviewed the file's own
predicted transitions on video after it was submitted. Each transition was shown as a six-second
clip, the broadcast picture next to the tactical picture with the tracking drawn on it, and
answered with one of: a change of that kind happens within one second of the prediction, it happens
at another moment, nothing of the kind happens, a different change happens, or cannot tell. The
answers are used to measure the file and to test variants of it. None of them enters the file as a
label; they chose one threshold, described below.

The reviewer's reading was checked first on the 15 predicted transitions inside the public window,
whose official answers are known. The two agree on 14. On the fifteenth the official labels give
the possession to the new team 2.3 s before the predicted moment and the reviewer saw no change.
In two more cases the reviewer accepted a change that the official labels place 1.4 s and 1.6 s
away, just outside the one-second tolerance, so the reviewer is, if anything, slightly lenient on
timing.

Over 168 reviewed transitions outside the public window (skips left out), the share that is real is:

| predicted change | reviewed | within 1 s | at any moment in the clip |
|---|---|---|---|
| team switch | 69 | 38% | 45% |
| ball goes dead | 49 | 57% | 67% |
| restart | 50 | 50% | 62% |
| all | 168 | 47% | 57% |

These figures are precision only, from one reviewer, with a 95% interval of about 8 points
overall and 11 to 18 points per kind of switch. They say nothing about real changes the file misses.
In the public window 13 of the 15 predicted transitions are real, so the public window looks like an
easier stretch than the rest of the match, and the full-match score may well be lower than the
public one; without a recall measurement this is not shown. A "no" covers both an invented change
and a real one more than 3 s away, outside the clip: in the public window two of the three team
switches answered "no" are real switches 4.6 s and 2.3 s earlier. The invented ones the reviewer
described are a block, an uncontrolled touch or a goalkeeper catch read as a touch or a stoppage, and
a pass that hands the ball over during a stoppage read as live play. Real team switches found at
another moment lay 2 to 3 s before the prediction in six of eight cases, but the clip reaches only
3 s back, so that range is partly its edge.

The answers were then used to test whether any of the uncalibrated settings removes invented
transitions without losing real ones: the team-switch cost, the live-to-dead and dead-to-live costs,
the control evidence gain, the contact distance and the walking-speed pivot of the liveness rule.
Of the 90 real transitions the submitted file keeps and the 68 invented ones, the best variant
removed 11 invented and lost 7 real ones while the public spellF1 fell from 0.607 to 0.552; the
best that left the public window unchanged removed 9 and lost 7, with one set piece fewer followed
by a restart. None was clearly better than the file before them, so none of these settings was changed. A
small logistic regression on the pipeline's own signals at each transition (spell lengths, posterior,
ball visibility and speed, distance of the new team to the ball, liveness, player speed, whistle,
team evidence) separates real from invented transitions only weakly, with a cross-validated AUC of
0.68 overall and 0.59 for team switches; with fifteen features on 62 switches that last figure is
mostly variance. Context the regression did not see separates better. A predicted stop with the ball
out of the pitch or a set-piece tag nearby is real in 24 of 28 reviewed cases, one without either in
13 of 25. The tracking features at the touch itself (relative speed, distance, hold length) carry
almost no signal on the reviewed switches, so telling a controlled touch from a block needs evidence
the tracking does not have.

The one pattern the review did isolate is a short team spell inside the other team's possession.
The file had ten such spells shorter than 3 s. All 20 of their edges were reviewed: 17 were answered "no",
2 "real" and 1 "at another moment", and every one of the ten spells has a "no" on its opening edge or
on both. Giving them back to the surrounding team removes 17 of the 80 reviewed invented transitions
and loses 2 of the 113 real ones, changes no public frame, keeps the set pieces at 58 of 62 and raises
phase-tag agreement from 82.6% to 83.4%. Spells of 3 to 5 s were mixed (one real, two invented, one
unclear), and the one reviewed spell of exactly 3.0 s was real, so the threshold stops below 3 s.

This choice has a known weakness, and we report it as found. The ground truth itself contains short
spells: the public window has a 0.8 s and a 1.3 s home spell inside away possession, which the file
misses. Shown those four transitions without being told, the reviewer accepted one and rejected three.
The official labels credit a team for a tackle until the next kick and for a single poke that wins
the ball; a human reviewer tends not to. So some of the spells the rule removes may be possessions
by the official convention, and the expected effect, a few hundredths of spellF1 either way, is not
established by the review. We kept the rule because on the reviewed evidence it removes nine
invented transitions for every real one it loses, and because it does not touch the public window.

## Limitations

The control signal cannot recover a possession the tracking does not show. Short duels and blocked
passes, where the corrected track behind the ground truth and the automatic track disagree about
who is at the ball, are the misses that remain. The broadcast tracking can only help where the two
views disagree, and on the public window it is right on 54% of those frames, barely more often
than the tactical view.

The whistle score does not change the decode. Stoppage length is set by the tracking gaps and by
player speed.

Liveness is rule-based. A gradient-boosted liveness model was built and is not used. Its training
labels were rules: dead when the ball is out of the pitch, during slow tracking gaps and just
before a kickoff; live when a fast ball is among running players. The model can tell those labels
apart from the very features that define them, so it repeats the rules everywhere else and calls
57% to 60% of graded time dead. For the same reason the decoder keeps its unnormalised team
emission, in which a live frame with no team evidence pays log 2 against dead. Normalising it moved
two of the four public stoppage onsets later and added possessions; under the learned liveness,
whose log-odds dwarf log 2, it changed almost nothing. A learned liveness needs labelled frames
where those rules say nothing.

The decoder can also give per-frame posterior probabilities by forward-backward on the same chain,
and a minimum-risk step can remove a predicted possession whose probability of being real is low.
Both are built and the removal is off. The posteriors are overconfident (the median predicted
possession averages 0.995 on its own label, and the one false possession in the public window
0.997). Fitted on the reviewed transitions, the probability that a possession is real does not
depend on its posterior at all, so the removal stays off.

The transition costs and the speed thresholds are physical values, not fitted. The reviewed
transitions show that moving them trades real transitions for invented ones at about the same rate
(previous section), and the sensitivity of an earlier file is in `analysis/sensitivity.md`. Moving
them to chase the public window is how the published baselines pick the wrong smoothing width.

What we would fix with more time follows from the review: a touch detector that tells a controlled
touch from a block or a deflection, stoppage detection that recognises a ball handed over at walking
pace, and a liveness model trained on reviewed frames rather than on its own rules.

## Reproduction

```bash
pip install -r requirements-pipeline.txt
python -m possession.make_submission --team <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

The match bundle belongs in `data/` as described in `DATA.md`, including the broadcast tracking
and its metadata. The command rewrites the CSV from that bundle. CPU only, deterministic.
`python -m pytest tests -q` checks the decoder, the broadcast offsets and that a fresh run matches
`tests/expected_sha256.txt`. Caches under `data/cache/` are derived match data and are not part of
the repository.

The reviewed transitions are in `validation/`. Copied to `data/labels/`, they rerun the variant test
(`python -m analysis.label_sweep`) and the classifier (`python -m analysis.transition_classifier`).
`python tools/review_web.py --name <name>` opens the review in a local browser.
