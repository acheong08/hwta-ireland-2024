# pyright: basic

from evaluation import evaluation_function
from utils import load_problem_data, load_solution

solutions = ["./output/1741.json", "./data/solution_example.json"]
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
