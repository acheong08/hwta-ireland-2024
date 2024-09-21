# pyright: basic

import os
import sys

from evaluation import evaluation_function
from utils import load_problem_data, load_solution

thedir = sys.argv[1] if len(sys.argv) == 2 else "output"
# List files in output directory
solutions = [
    f"./{thedir}/{f}" for f in os.listdir(thedir) if len(f) == 4 + len(".json")
]
solutions.sort(reverse=True)  # pyright: ignore[reportCallIssue]

for f in solutions:
    if not f:
        continue
    print(f"{f} - ", end="", flush=True)

    seed = 0
    fname = f.split("/")[-1]
    seed = int(fname.split(".")[0])
    fleet, pricing_strategy = load_solution(f)
    demand, datacenters, servers, selling_prices, elasticity = load_problem_data()
    score: int = evaluation_function(  # type: ignore[reportAssignmentType]
        fleet,
        pricing_strategy,
        demand,
        datacenters,
        servers,
        selling_prices,
        elasticity,
        seed=seed,
    )
    print(score)
