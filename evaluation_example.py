# pyright: basic

from evaluation import evaluation_function
from utils import load_problem_data, load_solution

solutions = [
    "./output/test.json",
    "./output/1741.json",
    "./output/3163.json",
    "./output/6053.json",
    "./output/2237.json",
]
for solution in solutions:
    # LOAD SOLUTION
    solution = load_solution(solution)

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices = load_problem_data()

    # EVALUATE THE SOLUTION
    score = evaluation_function(
        solution, demand, datacenters, servers, selling_prices, seed=1741
    )

    print(f"Solution score: {score}")
