"""E3 premise check: can the tactical camera's ground point be found from ball flights? (ROADMAP E3; nothing written)

    python -m analysis.e3_camera_point

If the ATD ball is the ground projection from one fixed camera, a ball in the air is reported displaced away from the
camera, so the reported path of a flight bulges off its ground chord, and the line through the point of largest bulge
along the bulge direction passes through the camera's ground point. Flights: visible runs faster than 12 m/s of 15-90
frames, chord over 15 m, bulge over 0.7 m in the middle half. The point is the least-squares intersection of those lines,
refitted three times on lines passing within 15 m with the bulge pointing away from it; 200 bootstrap resamples.
Then prints the geometry of the public 12665 header case.
"""
import numpy as np

from possession import ball, control, io, tracks

b_raw, players, team = io.load_atd()
players = tracks.repair_keepers(players, team); b = ball.repair(b_raw, players, team)
P = b["ball"]; print("x range", np.nanpercentile(P[:,0],[1,99]), "y range", np.nanpercentile(P[:,1],[1,99]))
acc = np.nan_to_num(b["acc"]); sp = np.nan_to_num(b["speed"]); vis = b["visible"] & ~b["oob"]
peaks = np.flatnonzero(control._local_peaks(acc, 15.0))
for _, lo, hi in io.halves():
    pts, dirs = [], []
    fast = (sp > 12) & vis; fast[:lo] = False; fast[hi:] = False
    e = np.flatnonzero(np.diff(np.r_[0, fast.astype(int), 0]))
    for a, c in zip(e[::2], e[1::2] - 1):
        if not (15 <= c - a <= 90):
            continue
        seg = P[a:c + 1]; A, L = seg[0], seg[-1]; ch = L - A; n = np.linalg.norm(ch)
        if n < 15: continue
        t = ((seg - A) @ ch) / n**2; foot = A + t[:, None] * ch; r = seg - foot; dev = np.linalg.norm(r, axis=1)
        k = dev.argmax()
        if dev[k] < 0.7 or not (0.25 < t[k] < 0.75): continue
        pts.append(seg[k]); dirs.append(r[k] / dev[k])
    pts, dirs = np.array(pts), np.array(dirs)
    M = sum(np.eye(2) - np.outer(u, u) for u in dirs); v = sum((np.eye(2) - np.outer(u, u)) @ p for p, u in zip(pts, dirs))
    C = np.linalg.solve(M, v)
    # robust: refit on lines passing within 15 m
    for _ in range(3):
        d = np.array([np.linalg.norm((np.eye(2) - np.outer(u, u)) @ (C - p)) for p, u in zip(pts, dirs)])
        # direction must point away from camera: (p - C) . u > 0
        ok = (d < 15) & (np.einsum('ij,ij->i', pts - C, dirs) > 0)
        M = sum(np.eye(2) - np.outer(u, u) for u in dirs[ok]); v = sum((np.eye(2) - np.outer(u, u)) @ p for p, u in zip(pts[ok], dirs[ok]))
        C = np.linalg.solve(M, v)
    print("half", lo, "flights", len(pts), "inliers", int(ok.sum()), "camera ground point", C.round(1), "median miss m", np.median(d[ok]).round(2))
    # bootstrap
    rng = np.random.default_rng(0); Cs = []
    idx = np.flatnonzero(ok)
    for _ in range(200):
        s = rng.choice(idx, len(idx)); M = sum(np.eye(2) - np.outer(u, u) for u in dirs[s]); v = sum((np.eye(2) - np.outer(u, u)) @ p for p, u in zip(pts[s], dirs[s]))
        Cs.append(np.linalg.solve(M, v))
    print("   bootstrap 90%", np.percentile(Cs, [5, 95], axis=0).round(1).tolist())

for fr in range(12640, 12672, 3):
    d = np.linalg.norm(players[fr] - P[fr], axis=1); d[np.isnan(d)] = np.inf
    order = np.argsort(d)[:3]
    print(fr, "ball", P[fr].round(1), "spd %.1f" % np.nan_to_num(b["speed"][fr]), "vis", int(b["visible"][fr]),
          [(int(j), int(team[j]), round(float(d[j]), 1), (players[fr, j] - P[fr]).round(1).tolist()) for j in order])
