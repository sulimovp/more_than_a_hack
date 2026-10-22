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

Shape model cross-half accuracy on self-labelled frames: 83.2% (half 1), 81.0% (half 2).
