"""Transition review in the browser: the same questions and answer file as tools/transition_review.py,
shown as clips with buttons. VALIDATION only; answers never feed the submission.

    python tools/review_web.py --name pavel            # then open http://127.0.0.1:8777

Each predicted transition plays as a 6-second loop: the broadcast picture (aligned to the tactical
timeline with broadcast.offsets()) on the left, the tactical picture with the tracking drawn on it on
the right, and a timeline with the predicted moment marked. One click answers; answers are written to
data/labels/transitions_<name>.csv after every click, in the format the scoring and fitting scripts
read. The server listens on 127.0.0.1 only: the clips are match video and must not leave this machine.
"""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from possession import broadcast, io  # noqa: E402
from transition_review import (CONTEXT, DEFAULT_CSV, answers_path, load_answers,  # noqa: E402
                               save_answers, transition_queue)

W, H_PX = 800, 450                       # each view
TEAM = {"H": "Barcelona", "A": "Athletic", "D": "dead ball"}
CACHE_SIZE = 6
PUBLIC_WINDOW = (11185, 18685)    # frames of the published ground-truth slice


class Clips:
    """Renders a transition's clip as JPEG frames (broadcast | tactical), cached, prefetched."""

    def __init__(self, queue):
        import cv2
        import load_homography
        self.cv2 = cv2
        self.queue = queue
        self.H, _, _ = load_homography.load()
        self.ball, self.players, self.team = io.load_atd()
        self.offsets = broadcast.offsets()
        self.halves = list(io.halves())
        self.tac = cv2.VideoCapture(io.asset_paths()["video"])
        self.bro = cv2.VideoCapture(os.path.join(io.DATA_DIR, "broadcast", "BAR-ATH_broadcast.mp4"))
        self.lock = threading.Lock()
        self.cache = {}

    def half_offset(self, frame):
        for half, first, stop in self.halves:
            if first - 2 * CONTEXT <= frame < stop + 2 * CONTEXT:
                return self.offsets[half]
        return None

    def _read(self, cap, start, n):
        out = []
        cap.set(self.cv2.CAP_PROP_POS_FRAMES, max(start, 0))
        for _ in range(n):
            ok, img = cap.read()
            out.append(img if ok else None)
        return out

    def _overlay(self, img, frame):
        from label_tool import _project
        cv2 = self.cv2
        m = self.H[frame]
        if np.isnan(m).any():
            return
        for k in range(self.players.shape[1]):
            x, y = self.players[frame, k]
            if np.isnan(x):
                continue
            px, py = _project(m, x / io.PITCH[0], y / io.PITCH[1])
            colour = (255, 255, 0) if int(self.team[k]) == 0 else (255, 0, 255)   # home cyan, away magenta
            cv2.circle(img, (int(px), int(py)), 10, colour, 3)
        bx, by = self.ball[frame]
        if not np.isnan(bx):
            px, py = _project(m, bx / io.PITCH[0], by / io.PITCH[1])
            cv2.circle(img, (int(px), int(py)), 7, (0, 230, 255), -1)

    def composite(self, i, size=(W, H_PX)):
        """(lo, hi, mark, [broadcast | tactical-with-tracking image per frame]) for queue item i."""
        cv2 = self.cv2
        w, h = size
        mark, _ = self.queue[i]
        lo, hi = max(mark - CONTEXT, 0), min(mark + CONTEXT, io.N_FRAMES)
        tac = self._read(self.tac, lo, hi - lo)
        off = self.half_offset(mark)
        bro = self._read(self.bro, lo - off, hi - lo) if off is not None else [None] * (hi - lo)
        images = []
        for k, f in enumerate(range(lo, hi)):
            left = cv2.resize(bro[k], (w, h)) if bro[k] is not None else np.zeros((h, w, 3), np.uint8)
            if tac[k] is not None:
                t = tac[k].copy()
                self._overlay(t, f)
                right = cv2.resize(t, (w, h))
            else:
                right = np.zeros((h, w, 3), np.uint8)
            images.append(np.hstack([left, right]))
        return lo, hi, mark, images

    def render(self, i):
        clip = self.cache.get(i)          # served without the lock, so a prefetch never blocks playback
        if clip is not None:
            return clip
        with self.lock:
            if i in self.cache:
                return self.cache[i]
            lo, hi, mark, images = self.composite(i)
            frames = [self.cv2.imencode(".jpg", im, [self.cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes() for im in images]
            clip = {"lo": lo, "hi": hi, "mark": mark, "frames": frames}
            self.cache[i] = clip
            for old in sorted(self.cache, key=lambda j: abs(j - i), reverse=True)[:max(0, len(self.cache) - CACHE_SIZE)]:
                self.cache.pop(old)
            return clip

    def prefetch(self, i):
        for j in (i + 1, i + 2):
            if j < len(self.queue):
                threading.Thread(target=self.render, args=(j,), daemon=True).start()


def question(trans):
    src, dst = trans.split(">")
    if src != "D" and dst != "D":
        return "{} loses the ball to {}?".format(TEAM[src], TEAM[dst]), \
            "Does {} take control of the ball from {} at the orange line?".format(TEAM[dst], TEAM[src])
    if dst == "D":
        return "Play stops ({} had the ball)?".format(TEAM[src]), \
            "Does the ball go out of play, or the referee stop play, at the orange line?"
    return "Play restarts with {}?".format(TEAM[dst]), \
        "Is the ball put back into play by {} (kick-off, throw-in, free kick, corner, goal kick) at the orange line?".format(TEAM[dst])


def other_answer(trans):
    src, dst = trans.split(">")
    if src != "D" and dst != "D":
        return "Something changes here, but play stops instead (ball out, foul)"
    if dst == "D":
        return "Something changes here, but {} wins the ball instead of play stopping".format(TEAM["A" if src == "H" else "H"])
    return "Play restarts here, but {} takes it".format(TEAM["A" if dst == "H" else "H"])


def make_handler(name, queue, clips):
    path = answers_path(name)
    answers = load_answers(path)
    page_path = os.path.join(HERE, "review_web.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            parts = self.path.split("?")[0].strip("/").split("/")
            if parts == [""]:
                return self._send(open(page_path, encoding="utf-8").read().encode(), "text/html; charset=utf-8")
            if parts[0] == "state":
                todo = [i for i, q in enumerate(queue) if q not in answers]
                return self._json({"n": len(queue), "done": len(queue) - len(todo), "next": todo[0] if todo else None})
            if parts[0] == "item":
                i = int(parts[1])
                mark, trans = queue[i]
                clip = clips.render(i)
                clips.prefetch(i)
                title, detail = question(trans)
                got = answers.get((mark, trans))
                return self._json({"i": i, "n": len(queue), "trans": trans, "title": title, "detail": detail,
                                   "other": other_answer(trans), "lo": clip["lo"], "hi": clip["hi"], "mark": mark,
                                   "fps": io.FPS, "answer": got[0] if got else None,
                                   "true_frame": got[1] if got else None})
            if parts[0] == "frame":
                clip = clips.render(int(parts[1]))
                return self._send(clip["frames"][int(parts[2])], "image/jpeg")
            self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            mark, trans = queue[int(body["i"])]
            ans = body["answer"]
            if ans not in ("Y", "N", "T", "E", "S"):
                return self._send(b"bad answer", "text/plain", 400)
            answers[(mark, trans)] = (ans, int(body["frame"]) if ans == "E" else None)
            save_answers(path, answers)
            self._json({"saved": len(answers)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--public-first", action="store_true",
                        help="put the transitions inside the public window first, to compare a labeller with the official labels")
    args = parser.parse_args()
    queue = transition_queue(args.csv)
    if args.public_first:
        first, stop = PUBLIC_WINDOW
        queue = sorted(queue, key=lambda q: not first <= q[0] < stop)
    done = load_answers(answers_path(args.name))
    queue = sorted(queue, key=lambda q: q not in done)   # answered clips first, so Back walks through them
    clips = Clips(queue)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.name, queue, clips))
    print("{} transitions; answers in {}; open http://127.0.0.1:{}".format(len(queue), answers_path(args.name), args.port))
    server.serve_forever()


if __name__ == "__main__":
    main()
