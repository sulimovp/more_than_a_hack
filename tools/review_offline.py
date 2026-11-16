"""Offline copy of the browser transition review, for a labeller on another computer.

    python tools/review_offline.py --name pavel --batch 40

Takes the clips that tools/review_web.py would still show (same order, answered ones left out), writes
each as a small H.264 file (broadcast | tactical with tracking, 6 s) and a self-contained page, in
batches of --batch clips under data/offline_review/ (gitignored: the clips are match video, for the
team's own labelling only). Each batch is also zipped. Answers are kept in the browser that opens the
page; its "Copy answers" button gives lines to paste back, which

    python tools/review_offline.py --name pavel --merge answers.txt

adds to data/labels/transitions_<name>.csv (the file the scoring and fitting scripts read).
"""

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from possession import io  # noqa: E402
from review_web import PUBLIC_WINDOW, Clips, other_answer, question  # noqa: E402
from transition_review import answers_path, load_answers, save_answers, transition_queue  # noqa: E402

OUT_DIR = os.path.join(io.DATA_DIR, "offline_review")
SIZE = (640, 360)          # each view; the clip is 1280 x 360


def pending_queue(name):
    queue = transition_queue()
    first, stop = PUBLIC_WINDOW
    queue = sorted(queue, key=lambda q: not first <= q[0] < stop)
    done = load_answers(answers_path(name))
    return [q for q in queue if q not in done]


def encode(path, images):
    import av
    with av.open(path, "w", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("libx264", rate=io.FPS)
        stream.width, stream.height = images[0].shape[1], images[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "33", "preset": "medium", "g": "10"}
        for im in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(im, format="bgr24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def build(name, batch, queue=None, label=None):
    """Batches of the pending queue, or of an explicit [(frame, transition)] queue written as <name>_<label>."""
    queue = pending_queue(name) if queue is None else queue
    clips = Clips(queue)
    template = open(os.path.join(HERE, "review_offline.html"), encoding="utf-8").read()
    n_batches = (len(queue) + batch - 1) // batch
    print("{} clips to review, {} batches of up to {}".format(len(queue), n_batches, batch), flush=True)
    for b in range(n_batches):
        folder = os.path.join(OUT_DIR, "{}_{}{}".format(name, label or "part", "" if label else b + 1))
        os.makedirs(os.path.join(folder, "clips"), exist_ok=True)
        items = []
        for i in range(b * batch, min((b + 1) * batch, len(queue))):
            mark, trans = queue[i]
            lo, hi, _, images = clips.composite(i, SIZE)
            clip = "clips/{}_{}.mp4".format(mark, trans.replace(">", ""))
            if not os.path.exists(os.path.join(folder, clip)):
                encode(os.path.join(folder, clip), images)
            title, detail = question(trans)
            items.append({"frame": mark, "trans": trans, "lo": lo, "hi": hi, "clip": clip,
                          "title": title, "detail": detail, "other": other_answer(trans)})
            print("part {} clip {}/{}".format(b + 1, i + 1, len(queue)), flush=True)
        manifest = {"name": name, "part": b + 1, "parts": n_batches, "fps": io.FPS, "items": items}
        page = template.replace("/*MANIFEST*/null", json.dumps(manifest))
        with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as handle:
            handle.write(page)
        shutil.make_archive(folder, "zip", folder)
        print("wrote", folder + ".zip", flush=True)


def merge(name, path):
    answers = load_answers(answers_path(name))
    added = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 4 or not parts[0].isdigit():
                continue
            frame, trans, ans, true_frame = parts
            if ans not in ("Y", "N", "T", "E", "S"):
                continue
            answers[(int(frame), trans)] = (ans, int(true_frame) if true_frame else None)
            added += 1
    save_answers(answers_path(name), answers)
    print("merged {} answers; {} in {}".format(added, len(answers), answers_path(name)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--merge", help="text file with the lines copied from the offline page")
    parser.add_argument("--clips", help="explicit list instead of the pending queue: frame:transition,... (e.g. 28940:H>A)")
    parser.add_argument("--label", default="check", help="folder suffix for --clips")
    args = parser.parse_args()
    if args.merge:
        merge(args.name, args.merge)
    elif args.clips:
        queue = [(int(c.split(":")[0]), c.split(":")[1]) for c in args.clips.split(",")]
        build(args.name, len(queue), queue, args.label)
    else:
        build(args.name, args.batch)


if __name__ == "__main__":
    main()
