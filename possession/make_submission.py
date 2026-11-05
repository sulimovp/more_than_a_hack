"""S7: the one command that regenerates the submission file.

    python -m possession.make_submission --team <team_name>

Writes evaluation/results/<team_name>.csv (gitignored), validates it with the official checks,
and prints its SHA-256 so a rerun can be compared byte for byte.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

from . import io, pipeline

import possession_metrics as pm  # noqa: E402  (io put evaluation/ on the path)

RESULTS = os.path.join(io.ROOT, "evaluation", "results")


def write_csv(codes, path):
    graded = io.graded_mask()
    if np.any(codes[graded] == io.ABSTAIN):
        raise ValueError("abstention on a graded frame: the decoder should label every one")
    labels = io.LETTERS[np.where(graded, codes, io.ABSTAIN)]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def codes_sha256(codes):
    """SHA-256 of the file write_csv() would write for these codes, for logging experiment runs."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        write_csv(codes, path)
        return sha256(path)
    finally:
        os.remove(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="possession_pipeline", help="file is named <team>.csv")
    parser.add_argument("--out", default=None, help="explicit output path")
    args = parser.parse_args(argv)

    path = args.out or os.path.join(RESULTS, args.team + ".csv")
    codes = pipeline.run()
    write_csv(codes, path)

    problems = pm.validate_submission(pm.read_label_csv(path), io.N_FRAMES)
    if problems:
        print("INVALID:", problems)
        return 1
    print("wrote", path)
    print("sha256", sha256(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
