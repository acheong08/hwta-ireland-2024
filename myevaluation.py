# pyright: basic

import json
import os
import subprocess
import sys

from evaluation import evaluation_function
from utils import load_problem_data, load_solution

thedir = sys.argv[1] if len(sys.argv) == 2 else "output"
# List files in output directory
solutions = [
    (f"./{thedir}/{f}" if f.endswith(".json") else None) for f in os.listdir(thedir)
]
solutions.sort()  # pyright: ignore[reportCallIssue]

solution_scores: dict[str, float] = json.load(open("solution_scores.json", "r"))

try:
    total_score = 0
    for f in solutions:
        if not f:
            continue
        md5sum = subprocess.run(
            ["md5sum", f],
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.split()[0]
        print(f"{f} - ", end="", flush=True)
        if md5sum in solution_scores:
            print(f"Score: {solution_scores[md5sum]}")
            total_score += solution_scores[md5sum]
            continue
        # LOAD SOLUTION
        solution = load_solution(f)

        # LOAD PROBLEM DATA
        demand, datacenters, servers, selling_prices = load_problem_data()

        seed = 0
        fname = f.split("/")[-1]
        if len(fname.split("_")) == 2:
            seed = int(fname.split("_")[0])
        else:
            seed = int(fname.split(".")[0])
        # EVALUATE THE SOLUTION
        score: int = evaluation_function(  # type: ignore[]
            solution,
            demand,
            datacenters,
            servers,
            selling_prices,
            seed=int(f.split("/")[-1].split(".")[0]),
        )

        solution_scores[md5sum] = score

        # get md5sum of the file

        print(f"{score}")
        total_score += score
    print(f"Average score: {total_score/(len(solutions))}")

except KeyboardInterrupt:
    pass
finally:
    json.dump(solution_scores, open("solution_scores.json", "w"))
