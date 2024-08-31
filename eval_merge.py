# pyright: basic

import os
import sys

from eval_utils import get_score

if len(sys.argv) != 3:
    print("Usage: python eval_merge.py <dir1> <dir2>")
    sys.exit(1)

# List files in output directory
solutions = [
    (f"./{sys.argv[1]}/{f}" if f.endswith(".json") else None)
    for f in os.listdir("output")
]

# Create merged directory if it doesn't exist
if not os.path.exists("merged"):
    os.makedirs("merged")

for f in solutions:
    if not f:
        continue
    seed = int(f.split("/")[-1].split(".")[0])
    print(f"{seed} - ", end="", flush=True)
    # ./<dir>/<seed>.json

    # EVALUATE THE SOLUTION
    score = get_score(f, seed)
    score2 = get_score(f"./{sys.argv[2]}/{seed}.json", seed)

    print(f"{score} vs {score2}")
    if score > score2:
        print(f"Better solution in {sys.argv[1]}")
        # Save the better solution to "./merged"
        os.system(f"cp {f} ./merged")
    elif score < score2:
        print(f"Better solution in {sys.argv[2]}")
        os.system(f"cp ./{sys.argv[2]}/{seed}.json ./merged")
    else:
        print("Same score")
        os.system(f"cp {f} ./merged")
