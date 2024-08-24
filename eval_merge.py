# pyright: basic

import os
import sys

from evaluation import evaluation_function
from utils import load_problem_data, load_solution

if len(sys.argv) != 4:
    print("Usage: python eval_merge.py <dir1> <dir2> <output>")
    sys.exit(1)

# List files in output directory
solutions = [
    (f"./{sys.argv[1]}/{f}" if f.endswith(".json") else None)
    for f in os.listdir(sys.argv[1])
]

# Create merged directory if it doesn't exist
if not os.path.exists(sys.argv[3]):
    os.makedirs(sys.argv[3])

for f in solutions:
    if not f:
        continue
    seed = int(f.split("/")[-1].split(".")[0])
    print(f"{seed} - ", end="", flush=True)
    # ./<dir>/<seed>.json
    # LOAD SOLUTION
    solution = load_solution(f)
    solution2 = load_solution(f"./{sys.argv[2]}/{seed}.json")

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices = load_problem_data()

    # EVALUATE THE SOLUTION
    score = evaluation_function(
        solution, demand, datacenters, servers, selling_prices, seed=seed
    )
    score2 = evaluation_function(
        solution2, demand, datacenters, servers, selling_prices, seed=seed
    )

    print(f"{score} vs {score2}")
    if score > score2:
        print(f"Better solution in {sys.argv[1]}")
        # Save the better solution to "./merged"
        os.system(f"cp {f} ./{sys.argv[3]}")
    elif score < score2:
        print(f"Better solution in {sys.argv[2]}")
        os.system(f"cp ./{sys.argv[2]}/{seed}.json ./merged")
