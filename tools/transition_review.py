"""Transition review: one question per predicted transition, for VALIDATION (ROADMAP.md E1). Answers never feed the submission.

The full match has about 430 predicted transitions. Reviewing each one as a 6-second clip measures
their precision directly, which the weak checks cannot, and is faster than labelling windows.

    python tools/transition_review.py --list                    # the queue, nothing to label
    python tools/transition_review.py --name alice              # review, resumes where you stopped
    python tools/transition_review.py --name bob --part 2/3     # split the queue between labellers
    python tools/transition_review.py --score                   # precision per type over every answer file
    python tools/transition_review.py --score --csv evaluation/results/other.csv

Each transition plays in a loop from 3 s before to 3 s after, with the tracking drawn on the video
and a bar marking the predicted frame. The kind of transition is shown (e.g. "H>A"), the rest of the
pipeline output is not. Keys:

    y         a transition of this kind happens within 1 s of the mark
    n         no transition here
    t         a transition, but to the other label (e.g. H>D instead of H>A)
    e         a transition of this kind, but more than 1 s away: pause (space), step to the frame
              where it really happens (, .) and press e there; that frame is saved
    s         skip (unsure)          b   back to the previous transition
    space     pause    , .  step one frame back / forward when paused
    [ ]       slower / faster        q   quit (answers save on every key)

Answers go to data/labels/transitions_<name>.csv (gitignored, like everything made from the video).
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
from possession import io  # noqa: E402
from gap_review import draw_bar, read_codes  # noqa: E402

LABEL_DIR = os.path.join(io.DATA_DIR, "labels")
CONTEXT = 3 * io.FPS
TOL = io.FPS                   # "within 1 s"
SEED = 2027                    # fixed queue order, so partial reviews still cover both halves
ANSWERS = {ord("y"): "Y", ord("n"): "N", ord("t"): "T", ord("e"): "E", ord("s"): "S"}
DEFAULT_CSV = os.path.join(ROOT, "evaluation", "results", "possession_pipeline.csv")


def letter(code):
    return {io.HOME: "H", io.AWAY: "A", io.DEAD: "D"}[code]


def kind(src, dst):
    if "D" not in (src, dst):
        return src + ">" + dst
    return "live>D" if dst == "D" else "D>live"


def transition_queue(csv_path=DEFAULT_CSV):
    """[(frame, 'H>A')] for every change of predicted label inside the graded blocks, shuffled."""
    codes = read_codes(csv_path)
    graded = io.graded_mask()
    idx = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    out = [(int(f), letter(codes[f - 1]) + ">" + letter(codes[f])) for f in idx if graded[f] and graded[f - 1]]
    order = np.random.default_rng(SEED).permutation(len(out))
    return [out[i] for i in order]


def answers_path(name):
    return os.path.join(LABEL_DIR, "transitions_{}.csv".format(name))


def load_answers(path):
    out = {}
    if os.path.exists(path):
        with open(path) as handle:
            next(handle)
            for line in handle:
                frame, trans, answer, true_frame = line.strip().split(",")
                out[(int(frame), trans)] = (answer, int(true_frame) if true_frame else None)
    return out


def save_answers(path, answers):
    os.makedirs(LABEL_DIR, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,transition,answer,true_frame\n")
        handle.writelines("{},{},{},{}\n".format(f, t, a, "" if tf is None else tf)
                          for (f, t), (a, tf) in sorted(answers.items()))


def score(answers_by_file):
    """Precision per kind: y counts as right; n and t as wrong; e as right-but-late (latency histogram)."""
    merged = {}
    for answers in answers_by_file:
        for key, (ans, tf) in answers.items():
            if ans != "S":
                merged.setdefault(key, []).append((ans, tf))
    rows = {}
    for (frame, trans), votes in merged.items():
        if len({v[0] for v in votes}) > 1:
            rows.setdefault(kind(*trans.split(">")), {}).setdefault("disputed", 0)
            rows[kind(*trans.split(">"))]["disputed"] += 1
            continue
        ans, tf = votes[0]
        r = rows.setdefault(kind(*trans.split(">")), {})
        r[ans] = r.get(ans, 0) + 1
        if ans == "E" and tf is not None:
            r.setdefault("late_s", []).append(round((tf - frame) / io.FPS, 1))
    return rows


def format_score(rows):
    if not rows:
        return "no transition answers yet (tools/transition_review.py)"
    lines = []
    for k in ("H>A", "A>H", "live>D", "D>live"):
        r = rows.get(k, {})
        n = sum(r.get(x, 0) for x in "YNTE")
        if not n:
            continue
        lines.append("{:7s} n={:3d}  precision@1s {:.2f}  within-any {:.2f}  wrong-kind {}  none {}  disputed {}{}".format(
            k, n, r.get("Y", 0) / n, (r.get("Y", 0) + r.get("E", 0)) / n, r.get("T", 0), r.get("N", 0),
            r.get("disputed", 0), "  late(s): " + str(sorted(r["late_s"])) if r.get("late_s") else ""))
    return "\n".join(lines)


def review(name, part, csv_path):
    import cv2
    import load_homography
    from label_tool import draw

    path = answers_path(name)
    answers = load_answers(path)
    queue = transition_queue(csv_path)
    k, n = part
    queue = queue[k - 1::n]
    H, _, _ = load_homography.load()
    b_raw, players, team = io.load_atd()
    cap = cv2.VideoCapture(io.asset_paths()["video"])
    todo = [i for i, q in enumerate(queue) if q not in answers]
    print("{} transitions in this part, {} answered, {} to go".format(len(queue), len(queue) - len(todo), len(todo)))
    pos = todo[0] if todo else len(queue)
    while pos < len(queue):
        mark, trans = queue[pos]
        lo, hi = max(mark - CONTEXT, 0), min(mark + CONTEXT, io.N_FRAMES)
        frame, playing, speed = lo, True, 1.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        key = None
        while True:
            shown = img.copy()
            draw(shown, frame, H, players, team, b_raw, None, playing, speed)
            cv2.putText(shown, "transition {}/{}  {}  {:+.1f} s  answer: {}".format(
                pos + 1, len(queue), trans, (frame - mark) / io.FPS, answers.get((mark, trans), ("-",))[0]),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            draw_bar(shown, frame, lo, hi, mark, mark + 1)
            cv2.imshow("transition review " + name, cv2.resize(shown, (1280, 720)))
            key = cv2.waitKey(max(1, int(40 / speed)) if playing else 0) & 0xFF
            if key in ANSWERS or key in (ord("b"), ord("q")):
                break
            if key == ord(" "):
                playing = not playing
            elif key == ord("["):
                speed = max(0.125, speed / 2)
            elif key == ord("]"):
                speed = min(8.0, speed * 2)
            elif not playing and key in (ord(","), ord(".")):
                frame = min(max(frame + (1 if key == ord(".") else -1), lo), hi - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, img = cap.read()
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
        ans = ANSWERS[key]
        answers[(mark, trans)] = (ans, frame if ans == "E" else None)
        save_answers(path, answers)
        pos += 1
    cv2.destroyAllWindows()
    save_answers(path, answers)
    print("saved", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name")
    parser.add_argument("--part", default="1/1", help="i/n: review every n-th transition starting at i")
    parser.add_argument("--list", action="store_true", help="print the queue")
    parser.add_argument("--score", action="store_true", help="precision per kind over saved answers")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    if args.list:
        queue = transition_queue(args.csv)
        kinds = {}
        for _, t in queue:
            kinds[kind(*t.split(">"))] = kinds.get(kind(*t.split(">")), 0) + 1
        print("{} transitions: {}".format(len(queue), kinds))
        for f, t in queue:
            print("  {:6d}  {}".format(f, t))
    elif args.score:
        files = sorted(glob.glob(os.path.join(LABEL_DIR, "transitions_*.csv")))
        print(format_score(score([load_answers(p) for p in files])))
    else:
        if not args.name:
            parser.error("--name is required to review")
        k, n = (int(x) for x in args.part.split("/"))
        review(args.name, (k, n), args.csv)


if __name__ == "__main__":
    main()
