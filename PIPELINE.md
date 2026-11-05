# Possession pipeline: how to run it, and where it stands

The design and its reasoning are in [PLAN.md](PLAN.md); the change-by-change record is in
[experiments.md](experiments.md); the submission note is [TECHNICAL.md](TECHNICAL.md).

## Run it

```bash
pip install -r requirements-pipeline.txt
python -m possession.make_submission --team <team_name>   # evaluation/results/<team_name>.csv, ~1 min
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
python evaluation/run_evaluation.py --bootstrap 1000      # public board, with the baselines
python -m possession.report                               # public unit tests + full-match weak checks
python -m pytest tests -q                                  # decoder cases + byte-for-byte reproduction
```

The match data goes in `data/` as the challenge README describes. Caches are written to
`data/cache/` (gitignored); delete it to rebuild from the raw files. CPU only, deterministic: the
reproduction test compares the file's SHA-256 with `tests/expected_sha256.txt`. If you change the
pipeline on purpose, regenerate the file and update that hash in the same commit.

## What each stage does

| stage | module | what it reads | what it produces |
|---|---|---|---|
| S0 | `io.py` | ATD, Smart Tagging, metadata | arrays indexed by tactical video frame, metres |
| S1 | `ball.py` | ball track | stuck and teleport detections removed, dropouts under 1 s filled, Savitzky-Golay velocity and acceleration, out-of-pitch with line |
| S2a | `audio.py` | the video's audio | narrow-band 2.5-4.5 kHz whistle score per frame |
| S2 | `liveness.py` | S1, players, S2a, tags | log-odds that the ball is in play; bonuses for entering a stoppage (whistle, ball out, ball vanishes) and leaving one (strike past 5 m/s, ball reappears) |
| S2 v2 | `livemodel.py` | S1, players, S2a, tags | P(alive) from a LightGBM on weak labels, cross-half. **Off** (`pipeline.LIVENESS_MODEL`): it fails the E6 checks, D share 57-60% |
| S3 | `control.py` | S1, players | H-vs-A log-odds from the 1.5 m zone, plus the last touch held until the next one |
| S5 | `restarts.py` | S2, tags | restarting team: kickoff laws, tagged set pieces, otherwise the taker |
| S4 | `shape.py` | players, first-pass labels | H-vs-A from team shape in ball gaps; LightGBM trained on self-labelled frames of the other half |
| S6 | `decoder.py` | all of the above | Viterbi over H/A/D per half, minimum spell 8 frames, minimum stoppage 50 frames. `posteriors()` gives forward-backward marginals on the same chain; the E5 minimum-risk spell removal is **off** (`MIN_RISK = None`) until the E1 labels set it |
| S7 | `make_submission.py` | S6 | the CSV, X outside the graded halves, D before the second-half kickoff |

## Where it stands (2 Nov)

| | public spellF1 | public macroF1 | possessions |
|---|---|---|---|
| this pipeline | **0.506** | **0.888** | 17 vs 17 true |
| `baseline_smooth50` (published floor) | 0.326 | 0.621 | 26 |
| `baseline_smooth125` | 0.424 | 0.581 | 16 |

Full match: 313 possessions, 117 stoppages, 49.4% of graded time dead, 81.6% agreement with
the Smart Tagging possession phases, 57 of 62 set pieces and goals followed by a restart.
The 22 Oct bootstrap on the previous file was 0.35–0.67; public spellF1 has not moved, so that
band still applies.

**Public numbers check the pipeline, they do not rank versions of it.** None of the constants were
fitted to the public window; four structural changes were made after reading its failures, each
for a reason that holds on the full match too (see experiments.md).

Robustness on the 22 Oct file, before the touch latch ([analysis/sensitivity.md](analysis/sensitivity.md); re-run the script for the current one). Each constant was halved and doubled:
- **Plateau:** the dead-ball transition costs, both minimum durations, the gap speed pivot and the
  whistle bonus move public spellF1 by under 0.04.
- **Slope, needs labelled windows:** `liveness.SPEED_PIVOT` (1.4 gives 0.40), `ACTIVITY_GAIN`
  (0 gives 0.42) and `control.CONTACT_SCALE_M` (0.7 gives 0.44). These set how much visible-ball
  play is called dead and how close a player must be to count.
- **Invisible to the public window:** `decoder.SWITCH` and `control.EVIDENCE_GAIN` move the
  full-match possession count from 234 to 336 while public spellF1 stays within 0.04. Only labelled
  windows can set them.
- **The whistle bonus is not decisive yet:** it fires at 38 of 122 stoppage onsets, but setting it
  to 0 or 8 changes nothing, because stoppage length is decided by the per-frame evidence. It is
  kept as a hook for a learned liveness model.

## What is left, in order of expected value

**Read [ROADMAP.md](ROADMAP.md) first.** The 2 Nov review found that about a third of the public
ground truth's team switches are not where the automatic ball track puts the nearest player, and
it lists the experiments in order, each with a script and an acceptance rule.

1. **Hand-labelled validation windows** (PLAN.md 5.2). Without them nothing here can be tuned
   honestly. `tools/label_tool.py` is ready: labellers calibrate on the public window first
   (`--score`), then label 6-8 windows of 3-4 minutes across both halves. It needs the GUI build of
   OpenCV (`opencv-python`); the headless build cannot open a window.
2. **How much visible-ball time is dead.** The pipeline calls 16.6 of the 62 visible-ball minutes
   dead. The public window suggests the share is lower. The labelled windows settle it; the
   relevant constants are `liveness.VISIBLE_CLIP`, `SPEED_PIVOT` and `ACTIVITY_GAIN`.
3. **Short spells where the nearer player is the wrong team.** Five public possessions are still
   misses (8.1 s, 2.8 s, 0.8 s, 1.3 s, 3.5 s). On those frames the true team is never at the ball,
   or only while the ball is moving past at 7–10 m/s. Two further control rules (a flight-only
   latch, a looser dribble gate) were run and not kept: possessions rose and the misses stayed.
   This is the ceiling of the automatic track. Labelled windows say whether it is common.
4. **Restart timing**: the dead-to-live transition lands within about 25 frames of a strike or
   reappearance; the labelled windows will say whether a fixed lag beats the current window.
5. Optional, only if labelled errors point there: PathCRF emissions in live gaps, a WASB ball
   detector for the 2-4 minutes of live play without a ball, a learned liveness model (LightGBM
   on weak labels, features as in Lang et al. 2022).
