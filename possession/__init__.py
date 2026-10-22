"""Possession pipeline: tactical ATD, Smart Tagging and audio in, one H/A/D/X label per frame out.

Stages, one module each (see PLAN.md):
    io        loading and frame alignment
    ball      ball track repair and kinematics
    audio     whistle score from the tactical video's sound
    liveness  P(ball in play) per frame
    control   per-frame H-vs-A evidence from player-ball geometry
    shape     H-vs-A from team shape where the ball is missing
    restarts  restart frames and restarting team
    decoder   offline Viterbi over H/A/D with minimum durations
"""
