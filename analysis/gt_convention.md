# E2: ground-truth convention study on the public window (3 Nov 2026)

Every team switch and every ignored contact that `python analysis/gt_convention.py` lists in the public window
(11185-18685), classified from rendered frames (`python analysis/render_frames.py ...`, PNGs under `data/tmp_e2/`,
not in git). Six events were classified in the 2 Nov review (ROADMAP.md 1.3); the other twelve were read here.

Classes (ROADMAP.md E2): **a** first contact, **b** touch-then-action (the GT waits for a controlled action, or ignores
a block or lost dribble), **c** ATD ball error (airborne ball, false detection, ball lagging), **d** tracking error.

## Team switches (12)

| frame | switch | what the frames show | class |
|---|---|---|---|
| 11351 | A>H | aerial duel: the ball is in the air at head height between Barcelona 2 and Athletic 44 at 11345; Barcelona 2 heads it, GT H from 11355. The ATD cross is 3 m off, on the ground projection | c (airborne) |
| 11553 | H>A | Barcelona 11 plays the ball at 11515, it rolls loose for 1.5 s and Athletic 36 meets it at 11553. The ATD cross lags the real ball by about 2 m | a |
| 11624 | A>H | scramble in the Barcelona box: Barcelona 15 blocks at about 11590-11600, the GT stays A; it switches when Barcelona 14 controls at 11624 | b |
| 12665 | H>A | Athletic header, ball 20 m/s in the air (2 Nov) | c (airborne) |
| 13704 | H>A | Athletic 34 intercepts Barcelona 2's dribble at about 13690; the GT switches 0.6 s later, after 34's next action | b |
| 13785 | A>H | stationary false ball detection on empty grass (2 Nov) | c (false ball) |
| 14846 | H>A | loose ball after a Barcelona 8 / Athletic 31 tussle; Athletic 28 is first to it at 14846, and the GT switches on that frame | a |
| 15346 | A>H | Koundé / Athletic 31 tussle credited to Barcelona until 31's kick (2 Nov) | b |
| 15365 | H>A | same event, second edge | b |
| 15766 | A>H | Barcelona 15 pokes a loose ball; GT credits one touch (2 Nov) | b |
| 15799 | H>A | same event: Williams' recovery needs a second action (2 Nov) | b |
| 17293 | A>H | home track count drops to 10 exactly at Pedri's touch (2 Nov) | d (dropout) |

## Contacts the GT ignores (6)

| frames | what the frames show | class |
|---|---|---|
| 11420-11438 | Athletic 44 dribbles for 2 s, Barcelona 2 wins it back before any pass (2 Nov) | b |
| 11709-11723 | **Barcelona's goalkeeper holds the ball, but the ATD has the away keeper slot 45 on him** (slot 23 is a ghost 3 m away). GT H is right | d (keeper slot) |
| 12442-12452 | Athletic 44 challenges Barcelona 2, who shields and plays on; 44 never controls | b |
| 13689-13702 | Athletic 34's interception, the same event as the 13704 switch (the GT waits 0.6 s) | b |
| 15654-15684 | **Athletic's goalkeeper collects and kicks long, but the ATD has the home keeper slot 23 on him** (slot 45 frozen at 96.4, 35.6). GT A is right | d (keeper slot) |
| 15783-15798 | Williams on the loose ball, same event as 15766/15799 (2 Nov) | b |

## Answers

**1. Fractions.** Over the 12 switches: a 2, b 6, c 3, d 1. Over all 18 events: a 2 (11%), b 10 (56%),
c 3 (17%), d 3 (17%). Two of the six b switches are each the second edge of a short blip, so by distinct events b is
still the largest class (6 of 14), with c + d at 6 of 14 close behind.

**2. Does the GT credit a loose-ball touch that is lost within 1 s?** Mostly no. A block (11624), a challenge
(12442), a lost dribble (11420) and an interception before its follow-up action (13704) are not credited, or are
credited late. 15766 is the one exception (one poke credited to Barcelona). The working rule: **possession changes on
the new team's first controlled action** (a pass, a carry, a shot, a goalkeeper's hold), not on proximity and not on
a block or deflection. That supports the roadmap's suggestion that control evidence should switch on a *kick*
(acceleration peak at a player) rather than on proximity, and that sustained contact should weigh less than the latch.

## New finding: goalkeeper slots sit on the wrong keeper

Slots 23 (home keeper) and 45 (away keeper) defend the right goal on average (half 1: slot 23 in the left box 93%,
slot 45 in the right box 89%), but they jump to the other keeper in runs. Graded frames with a keeper slot in the
**opponent's** box: 17,850 (11.9 min), of which **1,324 have that slot within 3 m of the ball**, so the control
evidence points at the wrong team. The longest run is slot 45 in Barcelona's box for the first 59 s of the public
window (11185-12662). A keeper never enters the opponent's box in open play, so this is detectable from the data
alone. See ROADMAP.md E4b.

## Decision

The judge in ROADMAP.md says E5 first if b dominates, E3/E4 if c + d dominate. b dominates the switches, but the
keeper-slot repair (E4b) is a data fix of about an hour that touches 1,324 graded frames, so it goes first. Then E5,
using the "first controlled action" rule above, then E3.
