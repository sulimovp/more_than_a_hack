# Path to win: possession pipeline for More Than A Hack 2027, phase 1

Written 2026-10-21. Built on the challenge docs, the Cursor plan (`~/.cursor/projects/.../canvases/possession-pipeline.canvas.tsx`),
measurements on the tactical data (`analysis/`), and a literature review (section 7).

**Status (24 Oct):** stages S0–S7 are built in `possession/`, including the S3 touch latch.
Public spellF1 0.506 against the floor's 0.326, macroF1 0.888. How to run it, current numbers
and what is left: [PIPELINE.md](PIPELINE.md).

Deadlines: same team name for every member by **10 Nov**, submission instructions **14 Nov**, upload by
**17 Nov 23:59 CET**. Top 20 advance. There is no leaderboard and no feedback before then.

---

## 0. The bet in one paragraph

Decode each half offline as a three-state sequence (H, A, D) with Viterbi, using per-frame evidence from
four sources: **liveness** (is the ball in play: player speed, ball gaps, out-of-pitch, whistles,
set-piece geometry), **control** (which team touched the ball last: distance and ball acceleration, ELASTIC
style), **restarts** (who restarts after each stoppage: Smart Tagging plus the laws of the game), and **team
shape** (who has the ball when the ball is missing during live play). Choose between versions using a
hand-labelled validation set that the rules allow, plus full-match weak checks from Smart Tagging. Never
choose between versions on the public 5 minutes.

The largest single correction to the earlier plans: **in the full match, a missing ball usually means a
stoppage, not live play.** Section 2 has the numbers.

---

## 1. What the ranking metric rewards

`spellF1` matches each predicted possession (a maximal run of H or of A) one-to-one with a true
possession, by temporal IoU, and averages F1 over IoU 0.3 to 0.7. Consequences:

| failure | what it does to spellF1 | design response |
|---|---|---|
| flicker (an extra 0.3 s spell) | one false positive, and it splits a true spell, so the true spell can drop below IoU | minimum spell length of about 0.3 s; a duel is not a handover |
| missed stoppage between two spells of the same team | two true spells merge into one predicted spell; both can fail IoU | liveness is first-class, not an afterthought |
| invented stoppage inside a possession | one true spell splits in two | D needs positive evidence |
| every boundary late by 8 frames | 0.89 at best (EVALUATION.md) | offline decoding, switches dated to first contact |
| every boundary late by 1 s | 0.65 at best | never use a causal confirmation window |
| short spells deleted (0.76 s, 1.32 s in public) | each is a whole missed possession | no smoothing window, no multi-second minimum |
| `X` on graded frames | free on spellF1 inside a spell, but charged by macroF1 (the tie-break) | **never write X on a graded frame** |

Graded frames: 11185–90651 and 110808–190197. Frames 110808–113741 are graded and are all `D` (pre-kickoff
of the second half). Write `X` outside the graded blocks.

---

## 2. What the data actually says (all reproducible)

```bash
python analysis/cache_atd.py          # numpy cache of the ATD, gitignored under data/cache
python analysis/full_match_stats.py   # gaps, speed, out of pitch, stuck ball, set pieces
python analysis/whistle_probe.py      # audio whistle peaks against public transitions
```

| fact | number | consequence |
|---|---|---|
| ball visible on graded frames | 59.3% in each half (88% in the public window) | the public window is not representative |
| ball gap lengths | 17 short dropouts under 1 s, **one gap of 1–5 s in the whole match**, 132 gaps of 5 s or more covering about 43 min | the ATD hides the ball over long stretches on purpose; the gaps behave like a built-in dead-ball flag |
| player speed in gaps of 5 s or more | median 0.94–1.00 m/s, against 1.6 m/s with the ball visible; 80–85% of gap frames are under 1.4 m/s | most gap time is a stoppage |
| gaps by median player speed | under 1.2 m/s: 77 gaps, 31 min; 1.2–1.6: 37 gaps, 9.5 min; over 1.6: 18 gaps, 2.3 min | about 31 min are almost certainly dead, about 10 min are ambiguous, about 2 min are live |
| set pieces with the ball missing around the tag start (−5 s to +2 s) | 76% of 59 | the restart is often where the ball reappears |
| public gaps | 2 of the 4 long gaps (5 s or more) are stoppages (ball visible 7% and 20% of the spell); 2 are live (5.3 s and 9.3 s, inside one Athletic possession, players faster at 3.3 and 1.6 m/s) | gap and speed together separate them; the ball-missing AUC of speed is 0.93 on public |
| player speed separates live from dead | AUC 0.83 on all public frames | a strong but not sufficient liveness feature |
| stoppages with the ball visible | 2 of 4 public stoppages (a foul at 15889, a goal kick at 18027 after the ball left at x < 0) | liveness cannot be "ball missing" alone |
| out of pitch | 4.2% of visible graded frames; D begins on the frame the ball crosses the line (18027) | high precision; date the onset by the crossing |
| GT handover timing | the switch is at first contact, about 1 m (14846, 11624), not after seconds of control | backdate every switch to the onset of contact |
| short public spells | 0.76 s and 1.32 s, both duels | allow spells of about 0.3 s and up |
| restart timing | GT D ends 10–12 frames after the ball reappears (12979→12989, 17826→17838; two samples) | hypothesis to check on the labelled set, not a constant to hard-code |
| stuck ball | 9 runs, 1,061 frames where the ball sits still for 2 s or more with nobody within 5 m (example 13785) | false detections; delete them before computing control |
| audio | the tactical MP4 has a 48 kHz AAC track aligned with the video (7,973 s) | whistles are usable |
| whistle peaks (2.5–4.5 kHz narrow-band) | 1–35 frames (at most 1.4 s) before 3 of 4 public stoppage onsets, and 8 frames after one restart; also some false peaks (crowd) | one liveness feature, never a rule on its own |
| Smart Tagging team tags | PROGRESSION, BUILD UP, LONG BALL, ATTACKING TRANSITION agree 73–100% with the GT team; DEFENDING… codes name the **defending** team; BALL IN FINAL THIRD and PLAYERS IN THE BOX are about 50/50 | weak labels only after inverting the defending codes and dropping the zone codes |
| Smart Tagging set-piece start | 3–12 s before the actual kick in public | look for the kick after the tag start, not at it |

**Correction to earlier reviews:** the Cursor plan said "long gaps are not stoppages, keep the last
team", and my first review guessed gaps were about 50/50. On the full match, about three quarters of gap
time is clearly a stoppage. Holding the last team through gaps would merge many possessions.

---

## 3. Architecture

```
            tactical ATD ──► S0 cache ──► S1 ball repair ──┬──► S3 control evidence (H/A log-odds)
                                                          │
 audio ──► S2a whistle score ──┐                          ├──► S4 team-shape model (H/A in live gaps)
 Smart Tagging ──► S2b events ─┼──► S2 liveness (P alive) ┤
 player kinematics ────────────┘                          └──► S5 restart model (strike frame, team)
                                                                          │
                                   S6 offline decoder (Viterbi, min durations, per half) ◄─┘
                                                │
                                   S7 writer: X outside graded, D 110808–113741, validate, SHA-256
```

Every stage writes a per-frame array to `data/cache/` (gitignored) so you can rerun and ablate each stage
on its own. Everything runs on CPU, deterministically, in under 10 minutes.

Suggested repo layout:

```
possession/            the pipeline package
  io.py                ATD/tagging/audio loading, frame alignment (video = ATD frame_id - 1)
  ball.py              S1
  liveness.py          S2 (+ audio.py for S2a)
  control.py           S3
  shape.py             S4
  restarts.py          S5
  decoder.py           S6
  make_submission.py   S7, the one command that regenerates the file
analysis/              measurements behind every claim in the tech doc
tools/label_tool.py    the hand-labelling tool (section 5.2)
tests/                 alignment, determinism (hash of the output), decoder unit cases
```

---

## 4. Stages in detail

### S0. Cache and alignment
- Parse the ATD once into float32 metres (`analysis/cache_atd.py` already does this, in 17 s).
- Video frame = ATD `frame_id − 1`. Smart Tagging frame = `round(seconds × 25)`. Audio time = frame / 25 s.
- Attack direction per half comes from the ATD metadata: Barcelona attacks left→right in the first half.
- Test: 199,319 rows, the first-half kickoff at 11185, the second at 113742.

### S1. Ball track repair
1. **Stuck ball:** remove runs where the ball moves under 0.3 m in 1 s for 2 s or more with nobody within
   5 m. Treat them as missing.
2. **Teleports:** remove single-frame jumps faster than 40 m/s that return within 3 frames.
3. **Short dropouts:** fill gaps under 1 s by linear interpolation (there are 17). Never interpolate across
   gaps of 5 s or more.
4. **Kinematics:** Savitzky–Golay velocity and acceleration (window 7 frames, order 2), as in ELASTIC.
   Acceleration norm includes direction changes, which is what marks a touch.
5. **Out of pitch:** ball beyond a line by more than 0.5 m for 2 frames or more. The onset is the first
   frame past the line, with hysteresis. Also record which line (touchline, goal line left/right): it
   tells you the restart type.

Output: `ball_xy`, `ball_vel`, `ball_acc`, `visible`, `oob`, `oob_line`.

### S2. Liveness: P(ball in play) per frame
This is the most valuable stage. Possessions are split by stoppages, and about 40% of graded time is
dead.

**Features** (each also taken as centred and lagged window means at ±0.4, 1, 2, 4 and 10 s, following
Lang et al. 2022, who reached 92% frame accuracy from players alone):
- collective player speed: median and 90th percentile; collective acceleration (a sharp drop marks the
  whistle)
- ball visible; position inside the current gap (elapsed and remaining time, gap length; offline, so the
  whole gap is known)
- ball out of pitch and the line it left by; distance to the nearest line and corner arc
- ball stationary with a player within 1 m (set-piece setup); ball speed
- whistle score (S2a), max over ±1 s
- Smart Tagging: time to the next and since the last set-piece or goal tag
- formation: players in each penalty box (corners, free kicks), both teams in their own halves (kickoff),
  the deepest player of a team holding the ball near the six-yard box (goal kick)

**S2a whistle score:** start with the narrow-band detector in `analysis/whistle_probe.py` (peak/broadband
ratio in 2.5–4.5 kHz). If it is too noisy, use PANNs CNN14 locally (AudioSet has whistle classes) as a
second score. Audio models must run locally: the terms forbid sending match data to hosted models.

**Model:** start rule-based (v1), then move to a small LightGBM or logistic model (v2).
- Training labels for v2 are weak labels on the full match, built from high-precision rules:
  - *dead*: out-of-pitch frames; gaps with median speed under 1.0 m/s; the 2 s before a kickoff formation
  - *live*: ball visible and moving with collective speed over 2 m/s; the 2 s after a confirmed kick
  - plus the public GT
- Keep the model small (about 15 features). Calibrate the probability (isotonic) on the hand-labelled set
  (section 5.2).

### S3. Control evidence: who touches the ball (ball visible only)
Per frame, a log-odds score `c(t)` for H against A:
- **Possession zone** of 1.5 m (Vidal-Codina et al. 2022; the databallpy default). One team in the zone
  pushes toward that team, weighted by `exp(-d/0.7 m)`.
- **Touch events** (ELASTIC): a local peak of ball acceleration over about 20 m/s², or a direction change
  over 10°, with a player within 1.5 m. The score jumps to that player's team **from the touch frame
  onward**. This is what dates switches correctly.
- **Duel:** both teams in the zone gives a small score toward the nearer player. The decoder's switch cost
  decides; the GT keeps 0.76 s duel spells, so do not suppress them completely.
- **Ball in flight:** ball speed over about 8 m/s and nobody within 2 m gives zero evidence. Hold the team.
- Ignore frames flagged as stuck or teleported.

### S4. Team-shape model: who has the ball when the ball is missing during live play
About 2–11 min of live play has no ball, and handovers happen inside those gaps.
- **Self-supervised labels:** frames with the ball visible, live (S2 above 0.9), and a player within
  1.0 m for 5 frames or more. The label is that player's team. That gives tens of thousands of frames and
  no hand labels.
- **Features** (oriented by attack direction): per team, centroid x and y, spread, defensive-line height,
  convex-hull area, mean velocity toward the opponent's goal, players in the opponent's half; the gap
  between centroids; 1 s deltas.
- **Model:** LightGBM. Train on half 1 and test on half 2, then the reverse. That is a full-match
  validation of this stage with no hand labels.
- **Use:** emissions inside live gaps; also a weak prior when the ball is visible but no one is near it.
- **Optional upgrade:** PathCRF (Kim et al., KDD 2026; MPL-2.0; pretrained weights for DFL tracking at
  5 fps) infers a ball-free possession path from player trajectories. Try it only as an extra emission in
  week 3, because it needs format adaptation and was trained on another league's data.

### S5. Restart model: when play resumes and who restarts
- **Strike frame:** after a dead stretch, the first frame where the ball (a) reappears and accelerates
  away from a player, or (b) starts moving from stationary at over 3 m/s, or (c) reappears with
  collective speed rising. Check the 10–12 frame "reappearance to GT restart" offset on the labelled set
  before using it.
- **Restart team, by type:**
  - kickoff at the start of the first half: the team that kicks off (Athletic at 11185); the second half
    is kicked off by the other team (Barcelona at 113742; check on video)
  - kickoff after a goal: the team that conceded (the GOALS tag team is the scorer, so take the other team)
  - goal kick: the defending team of that goal (tag team; cross-check the deepest player)
  - corner: the attacking team (tag team; cross-check the corner the ball is at)
  - throw-in (tagged N/A every time): the team of the player nearest the ball at the strike, who stands
    outside the touchline; cross-check against the opponent of the last toucher before the ball left
  - free kick: the tag team; if absent, the team that does not commit the foul, then the first control
- These become strong emissions for the first ~1 s after the strike.

### S6. Offline decoder
- An HMM with expanded states to enforce minimum durations: H₁…H_k, A₁…A_k, D₁…D_m (count-up chains, with
  a self-loop on the last state). Use k ≈ 8 frames (0.3 s) and m ≈ 50 frames (2 s; public stoppages are
  at least 5.5 s). That is about 70 states × 80k frames per half: numpy, a few seconds.
- Emissions per frame:
  - `log P(alive)` from S2
  - plus, for H/A, `±c(t)` from S3 where the ball is visible
  - or the S4 log-odds in live gaps
  - plus the S5 restart-team term after a strike
- Transition costs are few and physically motivated: H↔A switch, live→D, D→live. Set the starting values
  from rates (for example, about 3 possessions per minute gives roughly log(1/500) per frame) and tune them
  **only** on the hand-labelled set.
- Because decoding is offline, a switch lands on the first frame of evidence. No lag.
- Unit tests: hand-built sequences for a pass, an interception, a duel with a 0.76 s spell, a foul, a
  throw-in, a missing-ball live gap, and a missing-ball stoppage.

### S7. Writer
- `X` outside 11185–90651 and 110808–190197; `D` on 110808–113741; no `X` on graded frames.
- Run `evaluation/validate_submission.py` on the file; store its SHA-256 in `tests/expected_hash.txt`.
  CI fails if the file changes without the hash being updated on purpose.
- Name the file `<team_name>.csv`.

---

## 5. Validation: how you decide, given no leaderboard

### 5.1 The public window is a set of unit tests, not a score
For every one of the 21 public segments, print true and predicted boundaries and IoU. Each failure needs a
named cause. Record public spellF1, but **never accept or reject a change because of it**: one possession
moves it by about 6 points.

### 5.2 Hand-labelled validation windows (the main selection tool)
RULES.md: *"Small amounts of hand-labelled data used to train or validate a model are fine and normal."*
The submitted labels must still come out of the algorithm.
- Build `tools/label_tool.py`: an OpenCV player over the tactical video with the tracking overlaid through
  the homography. Keys H/A/D mark a change, with frame stepping. It writes `frame,label` change points.
- **Calibrate the labellers first:** each labeller labels the public window blind, then scores themselves
  with `run_evaluation.py`. The GT comes from corrected tracking ("dead" and "owner" as the tracking
  defines them), so learn its conventions: a switch at first contact, short duel spells, and when D starts
  and ends.
- Label about 20–30 min of new play: 6–8 windows of 3–4 min across both halves. Choose them to include
  gap-heavy play, set pieces, throw-ins and the goals. Have two labellers per window and settle the
  disagreements.
- Use them for **model selection and probability calibration only**. Do not train emission models on them,
  so that no hand label is ever copied into the output. Keep them out of git (they are not needed to
  regenerate the file) and say exactly how they were used in the tech doc. If in doubt, open an issue on
  the challenge repo and ask. The rules invite that.

### 5.3 Full-match weak checks (no labels needed)
- **Set-piece recall:** each of the 59 set pieces and 4 goals should have a D→live transition between the
  tag start and 15 s after it.
- **Phase-tag agreement:** the predicted team against PROGRESSION, BUILD UP, LONG BALL and the transition
  codes, plus DEFENDING… inverted.
- **Shape model:** cross-half accuracy (S4).
- **Plausibility:** possessions per half, D share of graded time (top-flight effective playing time is
  typically 50–60% of the clock, so expect roughly 40–50% D), share of spells under 1 s, distribution of stoppage
  lengths. Compare these against your own labelled windows, never against the 17 public possessions.

### 5.4 Decision rule
Accept a change only if spellF1 on the labelled windows improves by more than its bootstrap noise (resample
possessions with `--bootstrap 1000`), and no weak check gets materially worse. Otherwise keep the simpler
version. Log every run in `experiments.md`: git hash, parameters, labelled-window spellF1 and macroF1,
weak checks, and how many frames and spells changed against the previous run.

---

## 6. Schedule and roles (team of 3–5)

Roles:
- **R1 decoder and integration**: S6, S7, the single command, determinism
- **R2 liveness and restarts**: S2, S2a audio, S5
- **R3 control and shape**: S1, S3, S4
- **R4 validation**: label tool, calibration, labels, eval report, `experiments.md`
- **R5 (optional) video**: ball detector or PathCRF trial in week 3, then the tech doc and the clean-room
  reproduction

| week | build | gate to pass before moving on |
|---|---|---|
| **1: 21–26 Oct** | S0, S1; S2 v1 (rules: gaps + speed + OOB + pre-kickoff D); S3 v1 (zone + touches); S6 with min durations; S7. Label tool, labeller calibration on public, first 2 labelled windows | a valid full submission file exists (a safety net from day 5 on); public unit tests explained; clearly above `baseline_smooth50` on the labelled windows |
| **2: 27 Oct–2 Nov** | S2 v2 (features + whistle + LightGBM, weak labels); S5 restarts with team rules; touch timing; all 6–8 labelled windows done by 29 Oct | set-piece recall ≥ 90%; restart team right on the labelled windows; the decoder tuned on labelled windows only |
| **3: 3–9 Nov** | S4 shape model for live gaps; PathCRF trial; WASB ball detector on the tactical video **only if** error analysis shows that live-gap misses dominate. Confirm the team name for every member before 10 Nov | each addition has a written ablation; drop what does not move the labelled-window score beyond noise |
| **4: 10–17 Nov** | freeze code on 14 Nov; tech doc; clean-clone reproduction on a second machine; byte-identical file check; upload on **16 Nov**, not on the last day | `validate_submission.py` passes on the exact uploaded file; the hash matches a fresh run |

---

## 7. What the literature gives us

| work | what it is | how we use it |
|---|---|---|
| Lang et al., *Sci. Rep.* 2022, "Predicting the in-game status in soccer…" ([PMC9522646](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)) | live/dead from player positions only, 102 Bundesliga games; AdaBoost 92% frame accuracy, stoppage F1 0.80, median onset error about 1 s | the S2 feature design: multi-scale windows ±0.4…80 s, and past context weighs more than future context. Our extra signals (ball gaps, OOB, audio, tags) should do better |
| Vidal-Codina et al., *Sports Eng.* 2022 ([arXiv 2202.00804](https://arxiv.org/abs/2202.00804)) | rule-based events and possession from tracking; possession zone plus a ball speed or angle change marks a gain | S3 rules; defaults 1.5 m, 5 m/s, 10° (implemented in [databallpy](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html)) |
| ELASTIC, Kim et al. 2025 ([arXiv 2508.09238](https://arxiv.org/html/2508.09238v1), [code](https://github.com/hyunsungkim-ds/elastic)) | event and tracking sync; touch and receive detection from Savitzky–Golay ball acceleration peaks and player-ball distance | S1 kinematics and S3 touch detection (accel score saturating at 20 m/s², distance score to 3 m) |
| PathCRF, Kim et al., KDD 2026 ([arXiv 2602.12080](https://arxiv.org/html/2602.12080), [code](https://github.com/hyunsungkim-ds/pathcrf), MPL-2.0) | ball-free possession path from players only; Set Transformer plus a masked CRF and Viterbi; out-of-play as absorbing "outside" nodes; 90 min in under 1 min | the decoder design (constrained Viterbi, legal transitions); optional S4 emissions with pretrained weights (DFL data, 5 fps) |
| Kim et al., KDD 2023, "Ball trajectory inference from multi-agent sports contexts" ([arXiv 2306.08206](https://arxiv.org/abs/2306.08206)) | ball trajectory and player possession from player trajectories (Set Transformer plus Bi-LSTM) | background for S4; heavier than we need unless live gaps turn out to dominate |
| Link and Hoernig, *PLOS ONE* 2017 ([PMC5503225](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5503225/)) | individual possession with a Bayesian network over distance, direction, speed and acceleration; F 0.83 | supports soft per-frame control evidence rather than hard thresholds |
| WASB, Tarashima et al., BMVC 2023 ([arXiv 2311.05237](https://arxiv.org/abs/2311.05237), [code](https://github.com/nttcom/WASB-SBDT)) | strong sports ball detector and tracker, including soccer | the week-3 option for re-detecting the ball in live gaps; fine-tune on frames where the ATD sees the ball |
| T-DEED, Xarles et al., CVPRW 2024 ([code](https://github.com/arturxe2/T-DEED)); SoccerNet ball action spotting ([2024 results](https://arxiv.org/pdf/2409.10587), [2025 results](https://arxiv.org/pdf/2508.19182)) | precise pass and drive spotting in video; won SoccerNet ball action spotting 2024 | probably too costly here; only if video touch detection is needed for live gaps |
| PANNs CNN14 ([code](https://github.com/qiuqiangkong/audioset_tagging_cnn)); whistle detection by band energy ([arXiv 1411.6496](https://arxiv.org/pdf/1411.6496)) | pretrained audio tagging (527 AudioSet classes); classic 3.5–4.5 kHz whistle detectors | S2a, run locally |

---

## 8. Risks and mitigations

| risk | mitigation |
|---|---|
| tuning to 17 public possessions | section 5 decision rule; the public window is used for unit tests only |
| our labellers read "dead" and "owner" differently from the GT's corrected tracking | calibrate on the public window first; measure human-against-GT agreement and report it |
| the whistle detector fires on crowd whistling | it is one feature in S2, never a rule; require a speed drop together with it |
| a missed stoppage merges two possessions of the same team | the restart model gives an independent second signal (each restart implies a preceding D) |
| non-determinism breaks "same file" reproduction | CPU only; fixed seeds; LightGBM `deterministic=True` with fixed `num_threads`; pinned versions; hash test |
| hand labels seen as hand-correcting | validation and calibration only; never trained into emissions; documented; ask the organisers |
| match data leaks into the public repo | `data/` stays gitignored (the cache too); no frame-level exports, clips or labels committed |
| a team member's name mismatch disqualifies the team | everyone confirms the registered team name by 7 Nov |

---

## 9. Technical document outline (a few pages, written in week 4)

1. Approach: offline three-state decoding of liveness, control, restarts and shape, and why it beats
   smoothing.
2. Assets used and dropped: ATD, homography (label tool overlay), Smart Tagging (restarts, weak checks),
   audio (whistles); broadcast video dropped (unsynchronised, and 34% of its frames have no homography),
   with any other dropped trials and their ablations.
3. Possession rules as implemented: the switch at first contact, duels, loose balls, flight, D onset and
   end, restart team rules.
4. Validation: public unit tests, hand-labelled windows (how they were made, human-against-GT agreement),
   weak checks, bootstrap.
5. Limitations and what we would do next.
6. Reproduction: one command and the expected SHA-256.

---

## 10. Leave these alone

- Retuning a majority-vote window; any causal confirmation window (it adds lag).
- Copying Smart Tagging intervals as labels; using zone codes (FINAL THIRD, IN THE BOX) as team evidence.
- "Ball missing means D" with no speed or context, and also its opposite, "ball missing means hold the
  team".
- Targeting 17 possessions or the public mean duration.
- Broadcast synchronisation or a new tracker before week 3 error analysis asks for it.
- Sending any match data to a hosted model or API.
