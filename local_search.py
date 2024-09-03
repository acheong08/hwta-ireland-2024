import json

import numpy as np

import constants
import evaluation_v2
import generate
import reverse
from solver import models

servers = constants.get_servers()
datacenters = constants.get_datacenters()
selling_prices = constants.get_selling_prices()

for seed in [1741, 2237, 2543, 3163, 4799, 6053, 8237, 8501, 8933]:
    np.random.seed(seed)
    demand = constants.get_demand()

    def get_score(
        solution: list[models.SolutionEntry],
        seed: int,
    ) -> float:
        evaluator = evaluation_v2.Evaluator(
            solution, demand, servers, datacenters, selling_prices
        )
        return evaluator.get_score()

    initial_solution = reverse.get_solution(f"merged/{seed}.json")

    best_solution = initial_solution.copy()
    current_solution = initial_solution.copy()
    current_score = get_score(current_solution, seed)
    print("Initial score:", current_score)
    improved = True

    try:
        while improved:
            improved = False
            for entry in current_solution:
                # Try increasing the amount
                entry.amount += 1
                new_score = get_score(current_solution, seed)
                if new_score > current_score:
                    print("New score:", current_score)
                    current_score = new_score
                    improved = True
                    best_solution = current_solution.copy()
                    json.dump(
                        generate.generate(best_solution, servers),
                        open(f"output/{seed}.json", "w"),
                    )
                else:
                    # If increasing didn't help, try decreasing
                    entry.amount -= 2  # Subtract 2 because we added 1 before
                    if entry.amount < 0:
                        entry.amount = 0
                    new_score = get_score(current_solution, seed)
                    if new_score > current_score:
                        print("New score:", current_score)
                        current_score = new_score
                        improved = True
                        best_solution = current_solution.copy()
                        json.dump(
                            generate.generate(best_solution, servers),
                            open(f"output/{seed}.json", "w"),
                        )
                    else:
                        # If neither helped, revert to original
                        entry.amount += 1
    except KeyboardInterrupt:
        json.dump(
            generate.generate(best_solution, servers), open(f"output/{seed}.json", "w")
        )
