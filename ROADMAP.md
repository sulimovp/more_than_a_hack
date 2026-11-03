# Roadmap after the 2 Nov review

Written 2026-11-02 by an independent review of the pipeline in `possession/`. It is meant to be
run by another agent: every item names the files to touch, the command to run, and the rule for
keeping or dropping the change. Read [PIPELINE.md](PIPELINE.md) for how the pipeline runs and
[PLAN.md](PLAN.md) section 5 for the validation rules; both still hold.

Deadlines: team name by **10 Nov**, upload by **17 Nov 23:59 CET**. Freeze the code on 14 Nov.

---

## 0. The one-paragraph summary

The pipeline is sound, reproduces byte for byte, and its constants sit on a plateau. The review
fixed four small defects (section 2) without moving the public score. The real finding is in
section 1: **about a third of the ground truth's team switches are not where the automatic
tracking puts the nearest player**, and looking at the video frames shows three different causes,
only one of which is the ground truth's convention. The ceiling of "nearest player to the ATD
ball" is therefore around 12 of 17 public possessions, which is where the pipeline already is.
The experiments in section 3 attack those causes one by one; the first two are measurement, not
modelling, and nothing in section 3 should be kept on the public window alone.

---

## 1. What the review found

### 1.1 Verified

- `python -m possession.make_submission` reproduces the committed hash; `pytest` passes.
- The restart rules are right for tagged set pieces: all 12 tagged corners and all 7 tagged goal
  kicks sit on the correct side of the pitch for the tagged team, so `Team` on a `SET PIECES`
  tag is the taker. Kickoff laws match the nearest player on 5 of 6 kickoffs.
- Smart Tagging phase **ends** are near a true transition (7 of 8 unique ends in the public
  window within 1.2 s); phase **starts** are not (2 of 15). Full match: 283 phase ends, only 113
  of them within 2 s of a predicted transition.
- Decoder emissions: without normalisation, a frame with neutral team evidence pays log 2 against
  dead (P(H) = P(alive)/2). This is a real bias, but switching it off (`normalise_team=True`)
  makes three of the four public stoppage edges worse (D share 49.5% to 47.8%, public macroF1
  0.888 to 0.863). The bias is doing the work the liveness evidence should do at stoppage edges.
  Left off by default; see E6.

### 1.2 Fixed in this commit (public score unchanged, 678 graded frames moved)

| defect | where | what changed |
|---|---|---|
| the whistle bonus looked 1.5 s *ahead* (`maximum_filter1d` origin sign) although the whistle peaks sit 0 to 1.4 s *before* a stoppage | `liveness.evidence` | explicit look-back over the last 37 frames |
| a 90-degree "direction change" at the first frame of every visible ball run (angle against a missing neighbour), which registered a touch at every reappearance | `control._direction_change` | both frames must carry a velocity |
| the restart taker was the player nearest the ball in the 25 frames after the strike, right on 14 of 21 tagged set pieces (at a corner the ball is first seen in the box, among defenders) | `restarts.holder` | the player over the still ball in the 5 s before the strike (5 of 5 right where it answers), nearest player as a fallback |
| LightGBM on another CPU could flip a last bit and change the file | `shape.evidence` | P(home) rounded to 0.02 before it enters the decoder |

### 1.3 The ground-truth convention study (the finding that changes the plan)

`python analysis/gt_convention.py` lists the 12 team switches in the public window and the
contacts the ground truth ignores. `python analysis/render_frames.py <frames>` draws the frames.
What the frames show (home red, away blue, ATD ball yellow):

| case | frames | ATD says | video shows | cause |
|---|---|---|---|---|
| GT H>A mid-flight | 12650-12690 | nobody within 7 m, ball 20 m/s | an Athletic header at 12650, the ball lands with Athletic at 12690 | **airborne ball**: the ATD ball is the ground projection of a ball 2 m up, metres from the header |
| GT A>H | 13775-13804 | ball 7.5 m from everyone, 1 m/s | the yellow cross is on empty grass; the real ball is elsewhere | **false ball detection** that `ball.remove_stuck` did not catch (it drifts) |
| GT H blip 0.8 s | 15340-15370 | Athletic 31 nearer throughout | Koundé and Athletic 31 tussle; 31 kicks the ball at 15364; GT A from 15365 | **GT convention**: the tackle is credited to Barcelona until the kick |
| GT H blip 1.3 s | 15762-15802 | Athletic 33 (Williams) at 0.7 m | Barcelona 15 pokes the loose ball at ~15766, Williams takes it at 15778, GT A only at 15799 | **GT convention**: one touch wins possession, the recovery needs a second action |
| GT keeps H | 11420-11470 | Athletic 44 within 1 m for 2 s | Athletic 44 does dribble for about 2 s, Barcelona 2 wins it back at 11472 | **GT convention**: a dribble that is lost before a pass does not count |
| GT H 3.5 s | 17293-17320 | nobody within 2.3 m while the ball stops; then Koundé at 0.13 m, then 4.7 m | Pedri traps the ball at 17300 and Koundé plays it | **track dropout**: the home count dips to 10 exactly at the touch |

Consequences:

1. Short spells in the ground truth (0.8 s, 1.3 s) are not where the geometry says the other
   team is; they follow a touch-and-action rule that the ATD cannot reproduce. Chasing them with
   looser control rules (three such variants were tried on 24 Oct, one on 2 Nov) adds possessions
   everywhere else. **Do not add more control rules until E1 and E2 say which class dominates.**
2. Two of the six cases are the ATD ball being wrong in a way that is detectable from the data
   (ball in the air; a stationary detection far from every player). Those are E3 and E4.
3. Transition priors do nothing: a 2 to 4 nat bonus near Smart Tagging phase ends changed no
   frame at all, because on the misses the evidence has the wrong sign, not too high a cost.

### 1.4 Smaller observations, not acted on

- Pending restart (ball out until the next strike) holds the ball dead for 2.2 graded minutes
  while the ball is visible and inside the pitch; a throw-in received and dribbled under 5 m/s
  stays dead until the first pass. E2's transition review will show whether that is real.
- The pipeline calls 91% of missing-ball graded frames dead (the public window has 56%, but it
  is 88% visible and not representative) and 20% of visible-ball frames dead (public 15%).
- The deflection test in `control.touch_events` reads the relative speed *at* the acceleration
  peak, when a kicked ball is by definition leaving the kicker fast; the frame 15361 kick (home
  player at 0.8 m, peak 17.5 m/s², relative speed 8.0 at the peak but 4.7 five frames earlier)
  fails both the 20 m/s² threshold and the 8 m/s deflection rule. E4.
- A one-touch pass at 7 to 10 m/s relative speed is a possession in the ground truth and a
  "deflection" to the pipeline. E4.
- `tools/gap_review.py` (one answer per 5 to 20 s ball gap) was untracked; it is committed now.
  `python tools/gap_review.py --list` shows about 90 gaps, 15 minutes of video, half an hour of
  review. Nobody has answered one yet.
- The audio decode (`av`) could differ across ffmpeg builds; the whistle score is cached and only
  thresholded, and the bonus has never decided a frame, so the risk is small. Delete the cache
  and re-run on a second machine before the freeze (E8).

---

## 2. Decision rules (unchanged, restated so they are not skipped)

- The public window is unit tests. Record its numbers, never pick a version on them. One
  possession is about 0.06 spellF1 there.
- A change is kept if it (a) fixes a named failure on the public window **and** (b) does not
  move the full-match weak checks the wrong way (possession count, D share, set-piece restarts
  57/62, phase agreement 81.6%), **or** improves the labelled measurements from E1/E2 beyond their
  noise. Log every run in `experiments.md` with the hash.
- No hand label ever reaches the submission. Labels under `data/labels/` are for selection only.

---

## 3. Experiments, in order

**Status 3 Nov 2026.** E1: `tools/transition_review.py` is built and lists 428 transitions (H>A 100, A>H 98,
live>D 115, D>live 115); it needs people to watch the clips (about 45 min of video, split with `--part i/n`), so
the precision numbers are still open. E2: done, `analysis/gt_convention.md`. b (touch-then-action) is the largest
class; a new tracking defect (keeper slots on the wrong keeper, 1,324 ball-near graded frames) goes first as E4b.
**Order now: E1 labelling (team, in parallel), E5 with the "first controlled action" rule, E3, E6, E7.**
E5 rule part done 3 Nov (`control.confirmed`, 2 s): public spellF1 0.520 to 0.571, macroF1 0.915 to 0.919, possessions
293 to 271 (experiments.md). E5 steps 1-2 below (posteriors, minimum-risk spell removal) wait for the E1 labels, which
also have to confirm the 2 s window. **Order now: E1 labelling, E3, E5 steps 1-2 once labels exist, E6, E7.**

Each has: goal, what to build, how to run it, how to judge. Estimated effort is for one agent.

### E1. Transition review: precision of what the pipeline already predicts (half a day)

The cheapest labelled measurement there is. The full match has about 430 predicted transitions;
reviewing each as a 6-second clip is faster than labelling windows and measures precision
directly, which the weak checks cannot.

- Build `tools/transition_review.py` from `tools/gap_review.py`: queue = every predicted
  transition in the graded blocks (from `evaluation/results/possession_pipeline.csv`), clip from
  3 s before to 3 s after, keys `y` (a transition of this kind happens within 1 s), `n` (no
  transition here), `t` (a transition but to the other label), `e` (a transition, but more than
  1 s away: ask for the frame), `s` skip. Save to `data/labels/transitions_<name>.csv`. A
  `--score` mode reports precision per transition type (H>A, A>H, live>D, D>live) and the latency
  histogram for `e`.
- Split the queue among the team with `--part i/n` and shuffle with a fixed seed, as
  `gap_review.py` does, so partial reviews cover both halves.
- Judge: nothing to keep or drop; this produces the first honest precision numbers. Expect the
  live>D and D>live precision to be high (public: 4 of 4) and H<->A precision to be the weak one.
  Also run `python tools/gap_review.py` on the ~90 medium gaps: it gives recall on the
  live-gap-called-dead failure, which transition review cannot see.

### E2. Ground-truth convention study on the public window (half a day)

Before any new control rule, classify every public switch and every ignored contact into the
causes of section 1.3. Tooling exists: `analysis/gt_convention.py` lists them,
`analysis/render_frames.py` draws them. Write the result as a table in `analysis/gt_convention.md`
with one line per event. Then answer two questions in writing:

1. What fraction of public switches is (a) first contact, (b) touch-then-action, (c) ATD ball
   error, (d) track dropout? Six are classified above; twelve switches and six ignored contacts
   remain.
2. Does the ground truth ever credit a team for a loose-ball touch that is lost within 1 s? The
   15766 case says yes (Barcelona 15), the 11420 case says no (Athletic 44). If the answer is
   "a touch that moves the ball counts, a dribble that is dispossessed does not", the control
   evidence should switch on a *kick* (acceleration peak at a player) and not on proximity, and
   `TOUCH_HOLD_FRAMES`-style sustained contact should carry less weight than `LATCH_WEIGHT`.

Judge: the study decides which of E3, E4, E5 to run first. If (c)+(d) dominate, E3 and E4; if
(b) dominates, E5.

### E4 before E3 if E2 says so. E3. Airborne ball: distance to the camera ray (one day)

The tactical camera is fixed, high, at the halfway line on one side. The ATD ball is the ground
projection of whatever the detector sees, so a ball 2 m in the air is reported metres further
from the camera than the player under it (frame 12650 to 12665: the header is 7 m from the ATD
ball). The player under an airborne ball is close to the **ray** from the camera's ground point
through the reported ball position, not to the reported point itself.

- Camera ground point: project the image centre column downwards. Practical estimate: take the
  frames where the ball is struck from the ground at speed over 15 m/s and lands 2 s later
  (both ends on the ground, so both ATD positions are true), fit the reported mid-flight points
  against the straight ground path; the displacement direction at mid-flight points at the
  camera. One number per half (the camera does not move) in `games.py` or `possession/io.py`.
- In `control._closest`, when the ball is airborne (speed over 8 m/s, or the acceleration profile
  of a flight), replace `|player - ball|` by the distance from the player to the segment from the
  camera point to the ball, with a width of about 1.5 m. Keep the Euclidean distance otherwise.
- Judge: the 12665 header must become an Athletic touch at about 12650 (public unit test);
  possessions on the full match should not rise by more than the noise of E1's precision;
  `tools/transition_review.py --score` must not lose precision on H<->A.

### E4. Touch detector repairs (one day)

Three small changes to `control.touch_events`, run one at a time:

1. Deflection test on the relative speed **before** the peak (mean over the 5 frames before),
   not at it. A kick always reads fast at the peak.
2. `TOUCH_ACC` 20 to 15 with the pre-peak test from (1) in place.
3. Orphan touches: an acceleration peak over 15 m/s² with nobody within 1.5 m is a touch by an
   untracked player. Attribute it to the team of the slot that was within 3 m of the ball's
   position in the last second and is NaN now (frame 17300: Pedri's slot). If no such slot, leave
   it unattributed.

Judge: each must fix a named public case (15346 for 1 and 2, 17293 for 3), keep possessions
within noise, and keep or improve H<->A precision from E1. Log each in `experiments.md`.

### E4b. Goalkeeper slot repair (about an hour, first; added 3 Nov from E2)

**Done 3 Nov, kept**: public macroF1 0.888 -> 0.915, spellF1 0.506 -> 0.520, possessions 313 -> 293 (`experiments.md`).

Slots 23 (home keeper) and 45 (away keeper) jump onto the other team's keeper in runs: 17,850 graded frames
with a keeper slot in the opponent's box, 1,324 of them within 3 m of the ball (public cases 11709-11723 and
15654-15684; the first 59 s of the public window carry slot 45 in Barcelona's box). A keeper never stands in the
opponent's box in open play.

- In `possession/ball.py` or a new `possession/tracks.py`, after `io.load_atd()`: per half, using
  `io.home_attacks_right`, set a keeper slot to NaN on frames where it is inside the opponent's 18 m box
  (the other keeper slot is usually frozen or a ghost there; do not copy positions across).
- Run `python -m possession.make_submission`, then `pytest`.
- Judge: both public cases above stop being contacts for the wrong team; public macroF1 and possessions within
  noise or better; no change outside frames where a keeper slot was dropped (diff the CSVs). Log in `experiments.md`.

### E5. Spell-level decoding for spellF1 (two days, after E1)

Viterbi maximises frame-path likelihood; spellF1 scores possessions hit-or-miss at IoU 0.3 to
0.7. A wrong short spell costs a false positive *and* splits a true possession, so an uncertain
short spell should be dropped more readily than an uncertain long one. Two steps:

1. Forward-backward in `decoder.py` on the same state chain (the same emissions and transitions;
   sum instead of max) gives per-frame posteriors. Per predicted spell, record its mean posterior
   and its length; the E1 labels then give calibration: P(spell is real | posterior, length).
2. Minimum-risk post-processing: remove a predicted spell whose P(real) is under a threshold and
   merge its frames into the neighbouring label (the previous team, or D if both neighbours are
   D), then re-run the minimum-duration check. Choose the threshold on the E1 labels only.

Judge: expected spellF1 on the E1 labels; the public window must not lose a matched possession.
This is also where "chasing 0.8 s blips" is settled with numbers rather than by hand.

### E6. Liveness: the learned model and the emission bias (two days, after E1 and gap review)

PLAN.md S2 v2 was never built. With gap-review answers and E1's live<->D precision there is
finally something to fit against:

- Features per frame, as in Lang et al. 2022: collective speed and acceleration at windows of
  0.4, 1, 2, 4, 10 s; ball visible; gap elapsed and remaining; out of pitch; distance to the
  nearest line; ball speed; whistle max over 1 s; time to the next and since the last set-piece
  tag; players in each box; both teams in their own half.
- Weak labels on the full match: dead = out of pitch, the 50 frames before a kickoff, gaps with
  median speed under 1.0 m/s; live = ball visible and over 5 m/s with collective speed over
  2 m/s. LightGBM, small, cross-half as `shape.py` does, probability rounded as in `shape.py`.
- Then decide `decoder.viterbi(normalise_team=True)`: with a calibrated liveness the log-2 bias
  should no longer be needed, and the three public stoppage edges it protects should hold on
  their own. If they do not, keep the flag off and say so in TECHNICAL.md.

Judge: live<->D precision from E1, live-gap recall from the gap review, D share between 40% and
50%, set-piece restarts at least 57/62.

### E7. Restart timing and the pending hold (half a day, after E1)

E1's `e` answers give the latency of D>live transitions. The public window says the ground truth
turns live 10 to 12 frames after the ball reappears or 0 to 25 after a strike. If the latency
histogram has a clear mode, add it as a fixed lag on `leave_dead`. Also cap `pending_restart` at
a few seconds after the ball is back inside the pitch and visible if E1 shows throw-ins being
held dead too long.

### E8. Freeze protocol (14 to 16 Nov)

1. Fresh clone on a second machine, `pip install -r requirements-pipeline.txt`, empty
   `data/cache/`, `python -m possession.make_submission --team <team_name>`; the hash must equal
   `tests/expected_sha256.txt`. If the audio cache differs, compare `whistle_score.npy` and
   raise `WHISTLE_DB` if any frame sits within 0.5 dB of it.
2. `python evaluation/validate_submission.py` on the exact file to upload.
3. TECHNICAL.md: add the E2 table, the ablations from `experiments.md`, and how the labels from
   E1 and the gap review were used (selection only).
4. Upload on 16 Nov.

---

## 4. Ideas considered and parked

- **Smart Tagging phase ends as a transition prior**: no effect (section 1.3). Could still serve
  as a *feature* in E6's liveness model (time since the last phase end).
- **Smart Tagging team as per-frame evidence**: not tried; it would make the phase-agreement
  weak check circular, and the tags are phases, not possessions. Only with E1 labels to replace
  the check.
- **Latch reset at every stoppage**: 293 possessions and a worse public window; the restart team
  already handles it.
- **Broadcast ATD as a second ball**: 27% agreement with the public labels where it disagrees
  with the tactical ball (experiments.md). Parked.
- **Ball re-detection from the video (WASB)**: only if E2 shows ATD ball error dominates *and*
  E3 does not recover the airborne cases. A week of work; not before 10 Nov.
- **PathCRF emissions**: the shape model already gives 83% cross-half accuracy at zero cost and
  cannot overrule the latch; a stronger ball-free model only matters in live gaps, which are 2 to
  4 minutes of the match.

---

## 5. Do not

- Retune any constant on the public window. The sensitivity table shows which ones move it
  (`SPEED_PIVOT`, `ACTIVITY_GAIN`, `CONTACT_SCALE_M`); those wait for E1.
- Add control rules that make the nearest player win faster. Three were tried and each added
  20 to 30 possessions with the misses unchanged, and section 1.3 explains why.
- Commit anything under `data/`, including rendered frames and label files.
