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
| S0b | `broadcast.py` | broadcast ATD and metadata, tactical ball | the broadcast ball and players on the tactical timeline; one offset per half (4014, 26920) found from the two ball tracks and cached |
| S0c | `tracks.py` | players of either view | goalkeeper slots dropped while they stand in the opponent's penalty box |
| S1 | `ball.py` | ball track | stuck and teleport detections removed, dropouts under 1 s filled, Savitzky-Golay velocity and acceleration, out-of-pitch with line |
| S2a | `audio.py` | the video's audio | narrow-band 2.5-4.5 kHz whistle score per frame |
| S2 | `liveness.py` | S1, players, S2a, tags | log-odds that the ball is in play; bonuses for entering a stoppage (whistle, ball out, ball vanishes) and leaving one (strike past 5 m/s, ball reappears) |
| S2 v2 | `livemodel.py` | S1, players, S2a, tags | P(alive) from a LightGBM on weak labels, cross-half. **Off** (`pipeline.LIVENESS_MODEL`): it fails the E6 checks, D share 57-60% |
| S3 | `control.py` | S1, players, S0b | H-vs-A log-odds from the 1.5 m zone, plus the last touch held until the next one; a touch counts only as the new team's first controlled action (2 s), the reappearing ball is a touch by a player within 0.5 m, and a touch the broadcast contradicts (same ball within 3 m, other team's player within 1.5 m) is dropped |
| S5 | `restarts.py` | S2, tags | restarting team: kickoff laws, tagged set pieces, otherwise the taker |
| S4 | `shape.py` | players, first-pass labels | H-vs-A from team shape in ball gaps; LightGBM trained on self-labelled frames of the other half |
| S6 | `decoder.py` | all of the above | Viterbi over H/A/D per half, minimum spell 8 frames, minimum stoppage 50 frames. `posteriors()` gives forward-backward marginals on the same chain; the E5 minimum-risk spell removal is **off** (`MIN_RISK = None`) until the E1 labels set it |
| S7 | `make_submission.py` | S6 | the CSV, X outside the graded halves, D before the second-half kickoff |

## Where it stands (12 Nov)

| | public spellF1 | public macroF1 | possessions in the window |
|---|---|---|---|
| this pipeline | **0.607** | **0.930** | 12 vs 17 true |
| `baseline_smooth50` (published floor) | 0.326 | 0.621 | 26 |
| `baseline_smooth125` | 0.424 | 0.581 | 16 |

Full match: 259 possessions (median 8.4 s), 117 stoppages, 49.4% of graded time dead, 82.6%
agreement with the Smart Tagging possession phases, 58 of 62 set pieces and goals followed by a
restart. Submission sha256 `df2f5e7768aa…` (`tests/expected_sha256.txt`). The broadcast veto
added on 12 Nov changes no public frame; against the 4 Nov file it moves 1,071 graded frames,
possessions 274 to 259, phase agreement 81.8% to 82.6%. The 22 Oct bootstrap band on public
spellF1 was 0.35 to 0.67; one possession in the window is worth about 0.06.

**Public numbers check the pipeline, they do not rank versions of it.** None of the constants were
fitted to the public window; the structural changes were made after reading its failures, each
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

**Read [ROADMAP.md](ROADMAP.md) first.** No hand labels exist and none will come, so every item
that waited for the E1 transition labels or the gap review (E5 step 2, E6 retry, E7, the slope
constants) stays as it is.

Before the upload on 17 Nov:

1. **Second-machine reproduction** (ROADMAP E8 step 1). The fresh-clone dry run on this machine
   reproduces the hash (experiments.md, 5 and 12 Nov); a second machine has not been tried. A
   fresh run parses the broadcast ATD and searches the offsets (113 s in all on this machine, against 85 s before).
   If the hash differs, compare `whistle_score.npy` first, then the two broadcast caches.
2. **Team name in the file name**: `python -m possession.make_submission --team <team_name>`.
3. **`python evaluation/validate_submission.py`** on the exact file that is uploaded.

What remains open in the method, for a later phase: the five public misses are short spells the
ground truth credits to a touch the automatic tracking does not show as control, or spells where
the other team's player is nearest the ball in both views. The broadcast gives a second view of
those frames but, where the two views disagree, it is right only about half the time.
