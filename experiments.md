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
