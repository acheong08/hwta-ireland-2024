# pyright: basic

import os
import sys

from evaluation import evaluation_function
from utils import load_problem_data, load_solution

# List files in output directory
solutions = [
    (
        f"./{sys.argv[1] if len(sys.argv) == 2 else "output"}/{f}"
        if f.endswith(".json")
        else None
    )
    for f in os.listdir("output")
]

total_score = 0
for f in solutions:
    if not f:
        continue
    print(f"{f} - ", end="", flush=True)
    # LOAD SOLUTION
    solution = load_solution(f)

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices = load_problem_data()

    # EVALUATE THE SOLUTION
    score: int = evaluation_function(  # type: ignore[]
        solution,
        demand,
        datacenters,
        servers,
        selling_prices,
        seed=int(f.split("/")[-1].split(".")[0]),
    )

    print(f"{score}")
    total_score += score
print(f"Average score: {total_score/(len(solutions)-1)}")
