import copy
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

for seed in [3329, 4201, 8761, 2311, 2663, 4507, 6247, 2281, 4363, 5693]:
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
    # Preprocessing: manually add dismisses to the end of the lifespan
    for entry in initial_solution:
        if entry.action != models.Action.BUY:
            raise ValueError("Other actions not supported")
        # Split the dismiss action into smaller chunks for more fine tuned control
        NUM_CHUNKS = 10
        chunk_size = entry.amount // NUM_CHUNKS
        for _ in range(NUM_CHUNKS - 1):
            new_entry = copy.deepcopy(entry)
            new_entry.amount = chunk_size
            new_entry.timestep = entry.timestep + 96
            new_entry.action = models.Action.DISMISS
            current_solution.append(new_entry)
        # Put the remaining amount in the last chunk
        new_entry = copy.deepcopy(entry)
        new_entry.amount = entry.amount - (NUM_CHUNKS - 1) * chunk_size - 1
        new_entry.timestep = entry.timestep + 96
        new_entry.action = models.Action.DISMISS
        current_solution.append(new_entry)
    current_score = get_score(current_solution, seed)
    print("Initial score:", current_score)
    improved = True

    try:
        for entry in current_solution:
            if entry.action != models.Action.DISMISS:
                continue
            improved = True
            while improved:
                improved = False
                entry.timestep -= 1
                new_score = get_score(current_solution, seed)
                if new_score >= current_score:
                    print("New score:", current_score)
                    current_score = new_score
                    improved = True
                    best_solution = current_solution.copy()
                    json.dump(
                        generate.generate(best_solution, servers),
                        open(f"local/{seed}.json", "w"),
                    )
                else:
                    # Restore the previous state
                    entry.timestep += 1

    except KeyboardInterrupt:
        json.dump(
            generate.generate(best_solution, servers), open(f"local/{seed}.json", "w")
        )
