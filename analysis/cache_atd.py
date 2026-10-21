"""Parse the tactical ATD once into a numpy cache, so analysis scripts load in a second.

    python analysis/cache_atd.py          # writes data/cache/BAR-ATH_tactical_atd.npz

The cache sits under data/, which is gitignored: it is frame-level tracking, and publishing it
would redistribute the match data.

Arrays, indexed by video frame (ATD frame_id - 1):
    ball    (n, 2)      metres, NaN where the ball is missing
    players (n, k, 2)   metres per track slot, NaN where the slot is empty
    team    (k,)        0 home, 1 away
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(GAME))


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse(raw_path, n_frames, n_tracks, pitch):
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_tracks, 2), np.nan, dtype=np.float32)
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
    scale = np.array(pitch, dtype=np.float32)
    return ball * scale, players * scale


def load():
    if not os.path.exists(CACHE_PATH):
        build()
    data = np.load(CACHE_PATH)
    return data["ball"], data["players"], data["team"]


def build():
    paths = asset_paths()
    game = GAMES[GAME]
    team = team_of_each_track(paths["atd_meta"])
    ball, players = parse(paths["atd_raw"], game["n_video_frames"], len(team), game["pitch"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE_PATH, ball=ball, players=players, team=team)
    print("wrote", CACHE_PATH)


if __name__ == "__main__":
    build()
