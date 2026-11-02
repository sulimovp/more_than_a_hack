"""Gap review: one question per ball gap, for VALIDATION (PLAN.md 5.2). Answers never feed the submission.

The ball is missing on 41% of graded frames. Gaps over 20 s are almost all stoppages and the
pipeline calls them dead; the 5-20 s gaps (about 90 of them, 15 minutes) are the open question,
because each live gap called dead splits a possession in two. Answering them one by one is
about half an hour of work, much less than labelling whole windows.

    python tools/gap_review.py --list                      # the queue, nothing to label
    python tools/gap_review.py --name alice                # review, resumes where you stopped
    python tools/gap_review.py --name bob --part 2/3       # split the queue between labellers
    python tools/gap_review.py --score                     # pipeline vs every saved answer file
    python tools/gap_review.py --score --csv evaluation/results/other.csv

Each gap plays in a loop from 3 s before to 3 s after, with the tracking drawn on the video and
a bar showing where the gap is. Nothing from the pipeline is shown. Keys:

    h / a     live through the whole gap; home / away had the ball for most of it
    d         dead through the whole gap
    m         mixed: the ball went out or play restarted inside the gap
    s         skip (unsure)          b   back to the previous gap
    space     pause                  [ ]  slower / faster        q   quit (answers save on every key)

Answers go to data/labels/gaps_<name>.csv (gitignored, like everything made from the video).
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from possession import ball, io  # noqa: E402

LABEL_DIR = os.path.join(io.DATA_DIR, "labels")
MIN_S, MAX_S = 5.0, 20.0
CONTEXT = 3 * io.FPS
SEED = 2027                    # fixed queue order, so partial reviews still cover both halves
ANSWERS = {ord("h"): "H", ord("a"): "A", ord("d"): "D", ord("m"): "M", ord("s"): "S"}


def gap_queue(min_s=MIN_S, max_s=MAX_S):
    """[(start, stop)] of graded ball gaps between min_s and max_s, after S1 repair, shuffled."""
    b_raw, players, team = io.load_atd()
    b = ball.repair(b_raw, players, team)
    gaps = [(a, z) for a, z in ball.runs(~b["visible"] & io.graded_mask())
            if min_s * io.FPS <= z - a < max_s * io.FPS]
    order = np.random.default_rng(SEED).permutation(len(gaps))
    return [gaps[i] for i in order]


def answers_path(name):
    return os.path.join(LABEL_DIR, "gaps_{}.csv".format(name))


def load_answers(path):
    out = {}
    if os.path.exists(path):
        with open(path) as handle:
            next(handle)
            for line in handle:
                start, stop, answer = line.strip().split(",")
                out[(int(start), int(stop))] = answer
    return out


def save_answers(path, answers):
    os.makedirs(LABEL_DIR, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("start,stop,answer\n")
        handle.writelines("{},{},{}\n".format(a, z, v) for (a, z), v in sorted(answers.items()))


def all_answers():
    """{(start, stop): [answers]} over every labeller, skips dropped."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(LABEL_DIR, "gaps_*.csv"))):
        for gap, answer in load_answers(path).items():
            if answer != "S":
                merged.setdefault(gap, []).append(answer)
    return merged


def agreement(codes, answers=None):
    """Pipeline codes against the gap answers. None when nobody has answered yet.

    A gap counts as called dead when the pipeline labels most of it D. Gaps the labellers
    disagree on, and mixed gaps, only count towards n.
    """
    answers = all_answers() if answers is None else answers
    if not answers:
        return None
    res = {"n": len(answers), "live": 0, "live_called_dead": 0, "dead": 0,
           "dead_called_live": 0, "team_n": 0, "team_ok": 0, "disputed": 0}
    for (a, z), votes in answers.items():
        if len(set(votes)) > 1:
            res["disputed"] += 1
            continue
        answer = votes[0]
        pred = codes[a:z]
        called_dead = np.mean(pred == io.DEAD) > 0.5
        if answer == "D":
            res["dead"] += 1
            res["dead_called_live"] += int(not called_dead)
        elif answer in "HA":
            res["live"] += 1
            res["live_called_dead"] += int(called_dead)
            live = pred[pred != io.DEAD]
            if len(live):
                res["team_n"] += 1
                majority = io.HOME if np.mean(live == io.HOME) >= 0.5 else io.AWAY
                res["team_ok"] += int(majority == (io.HOME if answer == "H" else io.AWAY))
    return res


def format_agreement(res):
    if res is None:
        return "no gap answers yet (tools/gap_review.py)"
    return ("{n} gaps answered: live called dead {live_called_dead}/{live}, dead called live "
            "{dead_called_live}/{dead}, team in live gaps {team_ok}/{team_n}, disputed {disputed}"
            ).format(**res)


def draw_bar(img, frame, lo, hi, a, z):
    import cv2
    w = img.shape[1]
    x = lambda f: int((f - lo) / float(hi - lo) * (w - 20)) + 10
    y0 = img.shape[0] - 30
    cv2.rectangle(img, (10, y0), (w - 10, y0 + 16), (60, 60, 60), -1)
    cv2.rectangle(img, (x(a), y0), (x(z), y0 + 16), (0, 140, 255), -1)
    cv2.line(img, (x(frame), y0 - 6), (x(frame), y0 + 22), (255, 255, 255), 3)


def review(name, part):
    import cv2
    import load_homography
    from label_tool import draw

    path = answers_path(name)
    answers = load_answers(path)
    queue = gap_queue()
    k, n = part
    queue = queue[k - 1::n]
    H, _, _ = load_homography.load()
    b_raw, players, team = io.load_atd()
    cap = cv2.VideoCapture(io.asset_paths()["video"])
    todo = [i for i, gap in enumerate(queue) if gap not in answers]
    print("{} gaps in this part, {} answered, {} to go".format(len(queue), len(queue) - len(todo), len(todo)))
    pos = todo[0] if todo else len(queue)
    while pos < len(queue):
        a, z = queue[pos]
        lo, hi = max(a - CONTEXT, 0), min(z + CONTEXT, io.N_FRAMES)
        frame, playing, speed = lo, True, 1.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        key = None
        while True:
            shown = img.copy()
            where = "IN GAP" if a <= frame < z else ("before" if frame < a else "after")
            draw(shown, frame, H, players, team, b_raw, None, playing, speed)
            cv2.putText(shown, "gap {}/{}  {:.1f} s  {}  answer: {}".format(
                pos + 1, len(queue), (z - a) / io.FPS, where, answers.get((a, z), "-")),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            draw_bar(shown, frame, lo, hi, a, z)
            cv2.imshow("gap review " + name, cv2.resize(shown, (1280, 720)))
            key = cv2.waitKey(max(1, int(40 / speed)) if playing else 0) & 0xFF
            if key in ANSWERS or key in (ord("b"), ord("q")):
                break
            if key == ord(" "):
                playing = not playing
            elif key == ord("["):
                speed = max(0.125, speed / 2)
            elif key == ord("]"):
                speed = min(8.0, speed * 2)
            if playing:
                frame += 1
                ok, img = cap.read()
                if not ok or frame >= hi:  # loop the clip
                    frame = lo
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, img = cap.read()
        if key == ord("q"):
            break
        if key == ord("b"):
            pos = max(pos - 1, 0)
            continue
        answers[(a, z)] = ANSWERS[key]
        save_answers(path, answers)
        pos += 1
    cv2.destroyAllWindows()
    save_answers(path, answers)
    print("saved", path)


def read_codes(csv_path):
    import possession_metrics as pm
    return pm.label_codes(pm.read_label_csv(csv_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name")
    parser.add_argument("--part", default="1/1", help="i/n: review every n-th gap starting at i")
    parser.add_argument("--list", action="store_true", help="print the queue")
    parser.add_argument("--score", action="store_true", help="pipeline vs saved answers")
    parser.add_argument("--csv", default=os.path.join(ROOT, "evaluation", "results", "possession_pipeline.csv"))
    args = parser.parse_args()
    if args.list:
        queue = gap_queue()
        total = sum(z - a for a, z in queue) / io.FPS / 60
        print("{} gaps of {:.0f}-{:.0f} s, {:.1f} min of video".format(len(queue), MIN_S, MAX_S, total))
        for a, z in queue:
            print("  {:6d}-{:6d}  {:4.1f} s".format(a, z, (z - a) / io.FPS))
    elif args.score:
        print(format_agreement(agreement(read_codes(args.csv))))
    else:
        if not args.name:
            parser.error("--name is required to review")
        k, n = (int(x) for x in args.part.split("/"))
        review(args.name, (k, n))


if __name__ == "__main__":
    main()
