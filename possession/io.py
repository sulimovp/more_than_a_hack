"""Loading and frame alignment. Every array returned here is indexed by tactical video frame.

    video frame = ATD frame_id - 1
    video frame = round(Smart Tagging seconds * 25)
    audio time  = video frame / 25 s

Derived caches go to data/cache/, which is gitignored: they are frame-level match data and must
not be published.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, asset_paths, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH = GAMES[GAME]["pitch"]
HOME, AWAY, DEAD, ABSTAIN = 0, 1, 2, 3  # same codes as evaluation/possession_metrics.py
LETTERS = np.array(["H", "A", "D", "X"])

CACHE_DIR = os.path.join(DATA_DIR, "cache")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))


def cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, "{}_{}".format(GAME, name))


def halves():
    """[(half, first, stop)] graded blocks as half-open video-frame ranges."""
    return [(half, first, last + 1) for half, first, last in half_video_blocks(GAME)]


def graded_mask():
    mask = np.zeros(N_FRAMES, dtype=bool)
    for _, first, stop in halves():
        mask[first:stop] = True
    return mask


def team_of_each_track(meta_path):
    """0 for home, 1 for away, per track slot. The first team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def home_attacks_right(half):
    """True if the home team attacks towards x = PITCH[0] in this half."""
    root = ET.parse(asset_paths()["atd_meta"]).getroot()
    home = list(root.iter("Team"))[0]
    first_half_ltr = None
    for parameter in home.iter("ProviderParameter"):
        if parameter.findtext("Name") == "attack_direction_first_half":
            first_half_ltr = parameter.findtext("Value") == "left_to_right"
    if first_half_ltr is None:
        raise ValueError("attack direction missing from the ATD metadata")
    return first_half_ltr if half == 1 else not first_half_ltr


def _parse_atd(raw_path, n_tracks):
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    players = np.full((N_FRAMES, n_tracks, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, player_part, ball_part = line.rstrip().split(":")
            row = int(frame_id) - 1
            bx, by = ball_part.split(",")[:2]
            ball[row] = (float(bx), float(by))
            for k, group in enumerate(player_part.split(";")):
                if group[0] != "N":
                    px, py = group.split(",")[:2]
                    players[row, k] = (float(px), float(py))
    scale = np.array(PITCH, dtype=np.float32)
    return ball * scale, players * scale


def load_atd():
    """(ball (n, 2), players (n, k, 2), team (k,)), metres, NaN where missing."""
    path = cache_path("tactical_atd.npz")
    if not os.path.exists(path):
        paths = asset_paths()
        team = team_of_each_track(paths["atd_meta"])
        ball, players = _parse_atd(paths["atd_raw"], len(team))
        np.savez_compressed(path, ball=ball, players=players, team=team)
    data = np.load(path)
    return data["ball"], data["players"], data["team"]


def load_tags():
    import load_smart_tagging
    return load_smart_tagging.load(fps=FPS)


def load_gt_public():
    import possession_metrics as pm
    return pm.label_codes(pm.read_label_csv(GT_PUBLIC))
