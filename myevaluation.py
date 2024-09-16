# pyright: basic

import os
import sys

from eval_utils import get_score

thedir = sys.argv[1] if len(sys.argv) == 2 else "output"
# List files in output directory
solutions = [
    (f"./{thedir}/{f}" if f.endswith(".json") else "") for f in os.listdir(thedir)
]
solutions.sort()  # pyright: ignore[reportCallIssue]

total_score = 0
for f in ["output/123.json"] or solutions:
    if not f:
        continue
    print(f"{f} - ", end="", flush=True)

    seed = 0
    fname = f.split("/")[-1]
    if len(fname.split("_")) == 2:
        seed = int(fname.split("_")[0])
    else:
        seed = int(fname.split(".")[0])
    score = get_score(f, seed, rerun=True, fast=False)

    # get md5sum of the file

    print(f"{score}")
    total_score += score
print(f"Average score: {total_score/(len(solutions))}")
