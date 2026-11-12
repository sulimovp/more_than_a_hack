# Experiment log

One row per change that reached the decoder. "Public" is the 5-minute window (17 possessions;
about ±0.15 spellF1 is noise), used as unit tests only. "Phase" is full-match agreement with the
Smart Tagging possession-phase team tags (DEFENDING… inverted), a weak but match-wide signal.
"Set pieces" counts set-piece and goal tags followed within 30 s by a predicted restart.

Published floor for reference: `baseline_smooth50`, public spellF1 0.326, macroF1 0.621.

| date | change | public spellF1 | public macroF1 | possessions (full) | D share | set pieces | phase | kept? |
|---|---|---|---|---|---|---|---|---|
| 22 Oct | v1: gaps + speed liveness, nearest-player kernel with touch boost, Viterbi (k 8, m 50) | 0.418 | 0.804 | 429 | 45.6% | 52/62 (15 s window) | 75.6% | base |
| 22 Oct | team evidence as P(H \| alive), so it cannot make a stoppage look live; hold dead after the ball goes out until a strike; strike = speed crossing 5 m/s | 0.372 | 0.822 | 385 | 56.5% | 50/62 (15 s) | 77.8% | yes: all 4 public stoppages now matched; too many false stoppages left |
| 22 Oct | visible-ball floor −0.5, ±0.5 for a strike within 2 s, dead-ball transitions −15 | 0.449 | 0.849 | 389 | 53.3% | 49/62 (15 s) | 77.7% | yes |
| 22 Oct | control gated by player-ball relative speed (> 5 m/s counts 0.3), touch boost dropped | 0.509 | 0.862 | 275 | 53.6% | 49/62 (15 s) | 79.7% | yes: fewer flickers, phase agreement up |
| 22 Oct | S5 restart team (kickoff laws, tagged set pieces, taker) | 0.509 | 0.862 | 274 | 53.6% | 56/62 (30 s) | 79.8% | yes: principled, small effect |
| 22 Oct | S4 team shape in gaps, gain 0.25 | 0.482 | 0.845 | 281 | 53.4% | 56/62 | 80.5% | no: flipped a true 5.5 s Athletic stretch inside a live gap |
| 22 Oct | S4 team shape in gaps, gain 0.05 | 0.509 | 0.862 | 278 | 53.6% | 56/62 | 80.7% | yes |
| 24 Oct | S3 touch latch (acceleration peak, direction change, or the ball staying at a player's feet) and a capped ball-speed bonus on liveness | 0.506 | 0.888 | 313 | 49.5% | 57/62 | 81.5% | yes: dead-ball share is inside the 40–50% prior, and the four public stoppages rise from IoU 0.76–0.83 to 0.81–0.94. The 3.6 s Athletic spell now matches. Public spellF1 does not move. The five short misses remain: on those frames the nearer player is still the wrong team, or the touch is too brief to overcome the switch cost. The old false stoppage inside the long Barcelona spell is now a 1.6 s wrong-team spell |

Shape model cross-half accuracy on self-labelled frames: 83.2% (half 1), 81.0% (half 2).
Touches detected on 24 Oct: 1,095, about 10 per minute of graded time.

| 24 Oct | latch only while the ball is in flight, and a duel only when the two nearest players are within 0.4 m | 0.482 | 0.882 | 343 | 50.5% | 57/62 | 81.5% | no: 30 more possessions, public spellF1 down, the short misses unchanged |
| 24 Oct | sustained-contact relative speed 5 → 7 m/s | 0.506 | 0.887 | 346 | 49.6% | 57/62 | 81.0% | no: the 3.5 s miss stays dead, and possessions rise from 313 to 346 |
| 24 Oct | a local distance minimum under 0.45 m, with the other team at least 0.3 m farther, counts as a touch | 0.506 | 0.887 | 334 | 49.6% | 57/62 | 81.5% | no: 21 more possessions. The 3.5 s miss is a real meeting (0.13 m) but it sits inside a stoppage the decoder will not leave, because liveness there is about zero |

Two checks that did not reach the decoder. On visible flights in the public window the shape model and the touch latch each match the labelled team about 85% of the time; on the 470 frames where they disagree, each is right half the time, and the model is usually above 0.9 either way, so it cannot overrule the latch. The broadcast ball, aligned at tactical frame = broadcast frame + 4008, is a median 6.5 m from the tactical ball. Where they differ by more than 15 m, the team nearest the tactical ball matches the public labels 79% of the time and the team nearest the broadcast ball 27%.

| 2 Nov | review fixes: whistle window looks back instead of 1.5 s ahead; no 90-degree turn artefact at the first frame of a visible run; restart taker = the player over the still ball before the strike, nearest player only as a fallback; shape P(home) quantised to 0.02 | 0.506 | 0.888 | 313 | 49.4% | 57/62 | 81.6% | yes: 678 graded frames (0.4%) change, one fewer stoppage, one fewer spell under 1 s; public unchanged |

Tried on 2 Nov and not kept (ROADMAP.md section 1 has the numbers): team evidence normalised so that it cannot bias live against dead (`decoder.viterbi(normalise_team=True)`: D share 47.8%, three public stoppage edges worse); a transition bonus of 2-4 nats within 1.2 s of a Smart Tagging phase end (no change at all: the misses have evidence of the wrong sign, not too high a switch cost); latch reset at every stoppage (293 possessions, public spellF1 0.491).

3 Nov (no pipeline change): E1 tool `tools/transition_review.py` built (428 transitions queued; labels need people). E2 done, `analysis/gt_convention.md`: over 18 public events a 2, b 10, c 3, d 3; the GT moves possession on the new team's first controlled action, not on a block, a challenge or a lost dribble. New defect found: goalkeeper slots 23/45 sit on the other team's keeper in runs (17,850 graded frames in the opponent's box, 1,324 within 3 m of the ball). Next: E4b.

| 3 Nov | E4b goalkeeper slot repair (`possession/tracks.py`): per half, the keeper slot of each team (the slot most often in its own box: 23 home, 45 away in both halves) is set to NaN while it stands in the opponent's box; 17,850 graded keeper-slot frames dropped | 0.520 | 0.915 | 293 | 49.4% | 57/62 | 81.6% | yes: public macroF1 0.888 to 0.915, spellF1 0.506 to 0.520, 20 fewer possessions, stoppages unchanged. The false H at 15654 (home slot on Athletic's keeper) is gone and the opening A spell 11185-11351 is now found. Spell count in the window falls to 13 (truth 17): the A spell 14882-15928 now absorbs the 15766/15799 blip, which is a b-class GT convention case left to E5 |
| 3 Nov | E5 first controlled action (`control.confirmed`): a touch followed within 2 s (`CONFIRM_FRAMES` 50) by the other team's touch is dropped, so blocks, deflections and lost challenges no longer flip the latch. Sweep `python -m analysis.e5_controlled_action`: W 0/12/25/37 public unchanged (possessions 293/291/288/282), W 44/50/62 spellF1 0.571, macroF1 0.919 (possessions 274/271/268), W 75 the same public but 1,129 graded frames changed | 0.571 | 0.919 | 271 | 49.4% | 57/62 | 81.9% | yes: the public gain is one event, the false H/A spell at 12442 (Athletic 44's challenge, E2 class b) now absorbed into the long Barcelona spell; the plateau 44-62 is flat, stoppages unchanged, spells under 1 s 10. One public event is thin: confirm on the E1 labels (precision of H<->A before and after) before the freeze |

| 4 Nov | E4 touch repairs, sweep `python -m analysis.e4_touch_repairs` (switches in `control.py`, off unless noted). (1) deflection on the mean relative speed over the 5 frames before the peak: public unchanged, 277 possessions. (2) as (1) with `TOUCH_ACC` 15: spellF1 0.520, macroF1 0.915, 284. (3) orphan touches to a slot within 3 m in the last second and missing now: no change at all. None fixes its named case. 15346: at `TOUCH_ACC` 15 the home touch at 15360 (0.95 m, rel 7.4 m/s) is found, but E5 drops it because Athletic touches again at 15399; the 0.8 s H spell is the kind of event E5 suppresses, and the smoothed acceleration peaks 14 frames after the contact. 17293: the ball reappears at 17279 with home at 0.27 m and rel 15 m/s (the kick leaving the foot), a frame with no acceleration peak; orphan finds nothing because the ball was invisible over the look-back. Added (4) reacquisition touch (`REACQUIRE_M` 0.5): the first visible frame after 3+ invisible frames is a touch by the nearest player within 0.5 m, whatever the relative speed | 0.607 | 0.930 | 274 | 49.4% | 57/62 | 81.8% | (4) yes: 17293 H MISS to ok (IoU 0.70), the A spell before it 0.89 to 0.98, stoppages unchanged, spells under 1 s 10. 1.0 m scores the same (269 possessions); 0.5 kept as the narrower rule. (1)-(3) no. 15346 left: fixing it means a GT-convention exception to E5, which one public event cannot justify |

4 Nov, E3 (no pipeline change), `python -m analysis.e3_camera_point`: the camera ground point is not identifiable from flights. 17 and 18 fast flights per half (visible runs over 12 m/s, chord over 15 m, mid-flight bulge over 0.7 m); the bulge lines meet near the pitch centre ((55, 42) and (51, 39); bootstrap 90% boxes about 20 m by 15 m), with only 7 and 6 lines within 15 m of that point. A fixed camera beyond a touchline would put the point outside the pitch with most lines through it, so the bulges are swerve and noise, not projection. The named case does not need it either: on the repaired ball the header passes Athletic 40 at 0.9 m (12649) and Athletic 42 at 1.7 m (12658) at 19-22 m/s with no change of direction, so it is rejected by the 8 m/s deflection test, not by distance. E3 closed, FAIL as framed; the open lever for 12665 is a header-specific deflection rule, which one public case cannot set.

5 Nov, E5 step 1 (no pipeline change; submission sha256 `833f5c79b306…` unchanged), `python -m analysis.e5_posteriors`, written to `analysis/e5_posteriors.md`. `decoder.posteriors` runs forward-backward on the decoder's own chain: the same states, emissions and transition scores, log-sum-exp where Viterbi takes the max. It matches brute-force path enumeration to 1e-12 in `tests/`, and takes 10 s for the match. The posterior's most likely label equals the Viterbi label on 99.42% of graded frames; 1.83% of graded frames have no label above 0.9. Per predicted possession, the mean posterior of its own label is overconfident (5th / 25th / 50th percentile 0.926 / 0.985 / 0.995). It falls with length (median 0.979 under 1 s, 0.999 over 20 s), but the public window's one false possession (H 12801-12831, 1.2 s) sits at 0.997, so no calibration can be read off without labels. The per-spell table (start, stop, label, length, mean and minimum posterior, neighbours) is in `data/analysis/e5_spell_posteriors.csv` (gitignored, since it is the submission in another form).

5 Nov, E5 step 2 built and off (`decoder.MIN_RISK = None`; sha256 unchanged). `decoder.remove_uncertain_spells` removes the possession with the lowest P(real) = s(a + b logit(mean posterior) + c log(length s)) while that is under the threshold, one at a time. Its frames go to the previous team (the roadmap rule), or with `MIN_RISK_MERGE = "posterior"` to the neighbour the posterior prefers. Mechanics check with the placeholder calibration (a, b, c) = (0, 1, 0), not a selection:

| threshold | possessions removed | possessions (full) | D share | public spellF1 | public macroF1 | public matched lost | sha256 |
|---|---|---|---|---|---|---|---|
| off | 0 | 274 | 49.4% | 0.607 | 0.930 | - | 833f5c79b306 |
| 0.8 | 11 | 263 | 50.6% | 0.607 | 0.930 | none | fd9a16fff847 |
| 0.9 | 35 | 239 | 54.2% | 0.607 | 0.930 | none | 53e07eca1ce7 |
| 0.95 | 74 | 200 | 60.6% | 0.519 | 0.774 | A 16400-17293, H 17293-17380 | c05cca647475 |
| 0.98 | 135 | 139 | 70.4% | 0.519 | 0.774 | the same two | 682de878ec42 |

At 0.95 the 35 s A spell 16400-17279 (mean posterior 0.918) is removed and, after a stoppage, given to the next team. So the length term has to come from the labels, and D share rises because spells between two stoppages become D. Waiting for E1: `python -m analysis.e5_fit_min_risk`. It labels each predicted possession real or not from the answers on its two transitions, fits (a, b, c), and scores both merge rules at thresholds 0.05-0.95 by expected spellF1 on the labelled spells. It recommends a setting only if the 5th percentile of its bootstrap gain is above zero and the public window keeps every matched possession. Smoke-tested on random synthetic answers in a temporary folder outside `data/` (deleted); with no labels it prints that there is nothing to fit.

5 Nov, E6 (no pipeline change; `pipeline.LIVENESS_MODEL = None` and `normalise_team` off; sha256 unchanged), `python -m analysis.e6_liveness_model`. The weak-label liveness model as specified is in `possession/livemodel.py`:

- 22 features after Lang et al. 2022: collective speed and acceleration at 0.4-10 s, the gap features, the ball features, whistle, tag timing, box counts and own-half share.
- Weak labels: dead is out of pitch, the 50 frames before a kickoff, or a gap with median speed under 1.0 m/s; live is a visible ball over 5 m/s with players over 2 m/s. They cover 11.5% live and 19.4% dead of all frames.
- LightGBM as in `shape.py`, cross-half, P rounded to 0.02.

The labels are separable by the features that define them. Cross-half accuracy on labelled frames is 100.0% and 99.9%, and 95% of the gain goes to `line_m`, which is missing whenever the ball is. So the model learns that a missing or outside ball means dead: 0% of missing-ball graded frames are called alive, and the halves disagree on unlabelled frames (24% and 72% alive). As a diagnostic, the same model without the ball features (`livemodel.BALL_FEATURES`) learns the live label's own 2 m/s cut instead: 69% of the gain on 0.4 s speed, against a graded median of 1.35 m/s.

| variant (liveness, normalise_team) | public spellF1 | public macroF1 | possessions | stoppages | D share | D share ball visible / missing | set pieces | phase | under 1 s | frames changed | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rule, off (the submission) | 0.607 | 0.930 | 274 | 117 | 49.4% | 19.9% / 90.9% | 58/62 | 81.8% | 10 | 0 | 833f5c79b306 |
| rule, on | 0.579 | 0.917 | 282 | 117 | 47.9% | 18.8% / 88.9% | 58/62 | 81.3% | 11 | 2549 | 21c3016dd3f3 |
| model, off | 0.224 | 0.659 | 370 | 226 | 58.3% | 28.7% / 100% | 58/62 | 80.2% | 39 | 27416 | 4eecc506a5f9 |
| model, on | 0.224 | 0.659 | 368 | 224 | 58.2% | 28.6% / 100% | 58/62 | 80.3% | 37 | 27298 | 1d0a1d1b1246 |
| model + laws (out of pitch, pending restart), off | 0.230 | 0.672 | 351 | 214 | 59.7% | 31.2% / 100% | 57/62 | 80.5% | 33 | 25078 | 1c5c95206d8d |
| model + laws, on | 0.230 | 0.672 | 349 | 212 | 59.7% | 31.1% / 100% | 57/62 | 80.5% | 31 | 24954 | 7d345a17fb64 |
| rule + model log-odds, off | 0.277 | 0.698 | 364 | 217 | 57.2% | 26.8% / 100% | 57/62 | 80.5% | 34 | 25272 | fe45923605e1 |
| rule + model log-odds, on | 0.277 | 0.705 | 364 | 217 | 57.1% | 26.7% / 100% | 57/62 | 80.5% | 33 | 25148 | 4dea94f8869a |
| no-ball model, off | 0.180 | 0.626 | 361 | 272 | 71.4% | 55.7% / 93.5% | 57/62 | 82.3% | 39 | 40373 | 100fb0cb29e6 |
| no-ball model, on | 0.190 | 0.629 | 362 | 274 | 71.2% | 55.6% / 93.3% | 57/62 | 82.2% | 35 | 40142 | c8e65657a0fb |
| no-ball model + laws, off | 0.220 | 0.652 | 355 | 271 | 72.4% | 56.8% / 94.5% | 56/62 | 82.3% | 39 | 38699 | a7ba9abfdaa6 |
| no-ball model + laws, on | 0.220 | 0.654 | 357 | 273 | 72.2% | 56.6% / 94.3% | 56/62 | 82.2% | 37 | 38550 | 3616e29579cf |
| rule + no-ball model, off | 0.219 | 0.658 | 357 | 262 | 69.6% | 52.6% / 93.6% | 58/62 | 81.9% | 30 | 36498 | 42165a7e94fb |
| rule + no-ball model, on | 0.219 | 0.656 | 359 | 264 | 69.4% | 52.5% / 93.3% | 58/62 | 81.8% | 28 | 36379 | 79dec08e534d |

Public stoppage edges, predicted minus true start/stop in frames (IoU), for 12851-12989, 15889-16400, 17380-17838 and 18027-18389:

- rule, normalise_team off: −20/−10 (0.81), +39/0 (0.92), +24/+5 (0.94), −1/−23 (0.93).
- rule, normalise_team on: +30/−10 (0.71), +39/0 (0.92), +78/+5 (0.82), −1/−24 (0.93).
- The model variants start the 17380 stoppage 228-333 frames early and end 18027-18389 by up to 296 frames late.

Judge, label-free (ROADMAP E6 and section 2): FAIL, nothing kept.

- No model variant keeps D share within 40-50%. It is 57-60% as specified and 69-72% without the ball features. Set pieces stay at 56-58/62.
- `normalise_team=True` with the rule-based S2 passes D share (47.9%) and set pieces (58/62). But it fixes no named public failure, moves two weak checks the wrong way (possessions 274 to 282, phase agreement 81.8% to 81.3%), and makes two public stoppage onsets worse (12851 and 17380 above).
- With the model, `normalise_team` changes almost nothing, because the model's ±6.9 log-odds dwarf the log 2 bias.

Both flags stay off (TECHNICAL.md). The model needs labelled frames where the weak labels are silent: visible slow-ball play and gaps at 1.0-2.0 m/s. Those are exactly what the gap review and E1's live<->D answers give.

Set pieces: `python -m possession.report` on the committed file (sha256 `833f5c79b306…`, the 4 Nov E4 output) reads 58/62. The 4 Nov row says 57/62.

5 Nov, E8 step 1 dry run on this machine (no change; sha256 `833f5c79b306…` reproduced). Fresh `git clone` of `origin` at 82d4873 into a temporary folder, using the existing Python 3.13.5 with the pinned packages (nothing installed). The five tactical files were hard-linked into the clone's `data/tactical/`, not copied, and the cache started empty. `python -m possession.make_submission` rebuilt both caches and wrote the file in 85 s.

- The sha256 equals `tests/expected_sha256.txt`.
- Both rebuilt caches (`whistle_score.npy`, `tactical_atd.npz`) are byte-identical to the working copy's.
- The clone's `pytest` passes 22 of 22, `validate_submission.py` says valid, and the clone's `git status` is clean.
- The clone was deleted afterwards.

Heads-up for the second machine: 751 frames have a whistle score within 0.5 dB of `WHISTLE_DB` 13.0. Lowering the threshold by 0.5 dB changes nothing. Raising it by 0.25 or 0.5 dB moves the same 5 graded frames (69135-69140; sha256 `205ec86569a1` at 13.5 dB), with possessions and public scores unchanged. A hash mismatch there that comes from an ffmpeg build decoding the audio slightly quieter would be those 5 frames: compare `whistle_score.npy` first.

12 Nov, the broadcast tracking aligned (`possession/broadcast.py`, `python -m analysis.broadcast_alignment`; no change to the file by itself). The paragraph after the 24 Oct rows used tactical frame = broadcast frame + 4008 and found the two balls a median 6.5 m apart, and TECHNICAL.md called the broadcast unsynchronised. Both were wrong. The broadcast ATD has the tactical format and the same 0-1 pitch coordinates, and within a half the two recordings run at the same rate, so one offset per half aligns them. `broadcast.offsets()` derives it on the graded frames of each half: the share of frames with both balls visible that are within 3 m, over every shift that overlaps the half (every 4th shift on every 7th visible tactical frame, at least 10 minutes of frames with both balls), then every shift within 8 frames of the best; ties go to the smaller shift. It finds tactical frame = broadcast row + 4014 in half 1 and + 26920 in half 2 (34 s, cached in `data/cache/`; the test asserts both).

- At those offsets 0.800 and 0.782 of the 38,738 and 36,151 frames with both balls agree within 3 m, median distance 0.80 and 0.85 m. The best share more than 50 frames from the offset is 0.088 and 0.071.
- The best offset per 3-minute window lies within 3 frames of the half's offset in 18 of 18 and 17 of 18 windows (range -3 to +2 and -3 to +4), so replays and cut-aways do not move it.
- The first team in the broadcast metadata is home, as in the tactical file: with a player within 1.5 m of the ball in both views (keeper slots repaired in both; `tracks.repair_keepers` finds slots 25 and 50 on the broadcast), the two name the same nearest team on 93.4% of 24,617 frames, 94.7% where the balls agree.
- Public window: the broadcast ball is visible on 90.5% of true-live and 5.7% of true-dead frames. On 1,305 true-live frames with a player within 1 m in both views, the views agree on 94.5%; where they agree the truth matches on 98.1%, against 92.8% for the tactical view alone; where they disagree the tactical view is right on 45.8% and the broadcast on 54.2%. (The 12 Nov scratch count without the broadcast keeper repair read 93% and 57%.)

| date | change | public spellF1 | public macroF1 | possessions (full) | D share | set pieces | phase | kept? |
|---|---|---|---|---|---|---|---|---|
| 12 Nov | broadcast veto (`control.contradicted`, `python -m analysis.broadcast_control`): a confirmed touch is dropped when the aligned broadcast ball is within 3 m (`VETO_BALL_M`) of the repaired tactical ball and the broadcast's nearest player, within 1.5 m (`VETO_PLAYER_M`), is of the other team; 26 of 1,044 touches, one of them in the public window | 0.607 | 0.930 | 259 | 49.4% | 58/62 | 82.6% | yes: no public frame changes (every segment as before), stoppages 117 unchanged, 1,071 graded frames change, spells under 1 s 10 to 11, sha256 `df2f5e7768aa`. On the changed frames inside an unambiguous phase tag, 607 agree with the tag after and 190 before, but that is nine stretches (6 move towards the tags, 3 away), so the evidence is modest. Without the veto the same script reproduces `833f5c79b306`. With 2.5 m: 32 touches, 257 possessions, 82.8%, public the same (`2c4382df58a4`); 1.5 m kept as the narrower rule, not swept further |
| 12 Nov | broadcast stand-ins (`python -m analysis.broadcast_control`): where the balls agree within 3 m and the broadcast has a player of team T within 1.5 m of its ball while the tactical view has no T player within 3 m of that spot, the broadcast player is added as an extra tactical player (tactical ball plus its offset from the broadcast ball), and control runs on the augmented players | 0.635 | 0.937 | 261 | 49.4% | 58/62 | 82.4% | no: phase agreement falls below the veto-only file, and neither named case is fixed. 11351-11466: the H spell now starts at 11401 instead of 11466, so the A spell before it rises from IoU 0.59 to 0.77 (all of the public gain), but the H spell 11351-11553 is still a miss (0.06 to 0.11) and the long H spell after it falls from 0.83 to 0.79. 15766-15799: no stand-in at all. 194 graded frames change, stoppages unchanged, `9b612ae2981a`. Stopped here as planned |

Before the stand-ins, how often the situation occurs: of 28,442 graded frames with the balls within 3 m and a broadcast player within 1.5 m of its ball, the tactical view has no player of that player's team within 2 / 3 / 5 m of the tactical ball on 3,047 / 1,300 / 910 (10.7% / 4.6% / 3.2%; at 3 m 933 home, 367 away). It holds at 11 of the 1,044 confirmed tactical touches. In the public case 11351-11466 the balls agree on all 115 frames and it holds on 36; in 15766-15799 on 5 of 33. With the stand-ins, 1,287 graded frames (0.9 min) carry one, from 30 broadcast slots, and the confirmed touches go from 1,044 to 1,064 (8 gone, 28 new).

Tried on 12 Nov in scratch code and not kept (none of it is in the repository; do not retry):

- A broadcast-only LightGBM liveness model trained on the pipeline's own decode (public AUC live against dead 0.963; the tactical rules 0.983; their sum 0.989), added to the alive log-odds two-sided at gain 0.5, 1 and 2, or one-sided: public macroF1 falls to 0.90-0.92. TV replays run into restarts and the wide shot returns before a free kick is taken, so the stoppage 15889-16400 ends at 16526 or 16341 instead of 16400.
- `control.evidence` run on the broadcast view and combined with the tactical one by mean or by sum: public spellF1 0.510 and 0.423; the 17293 and 13704 spells are lost.
- Zeroing the tactical zone signal where the broadcast disagrees: 7 frames change, nothing else.
- Restarting a sustained-contact hold run when the nearer team changes: 24 such runs, public spellF1 0.566, phase agreement 81.4%.
- Handing a keeper slot found in the opponent's box to the opponent's keeper slot instead of dropping it: 145 positions, 2 frames change.

A one-off sanity check, not used to select anything (it leans on numbers derived from the private ground truth): scoring the four committed baselines against the 4 Nov file (`833f5c79b306`) as if it were the truth gives spellF1 0.051 / 0.197 / 0.229 / 0.245 for raw, smooth25, smooth50 and smooth125, against their published full-match 0.0486 / 0.1996 / 0.2433 / 0.2242.
