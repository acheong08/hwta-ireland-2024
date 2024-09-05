import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import constants
import evaluation_v2
import generate
import reverse
from solver import models
from utils import demand_to_map, sp_to_map

servers = constants.get_servers()
server_map = {s.server_generation: s for s in servers}
datacenters = {dc.datacenter_id: dc for dc in constants.get_datacenters()}
selling_prices = sp_to_map(constants.get_selling_prices())

stop = False


def optimize_seed(seed: int):
    global stop
    np.random.seed(seed)
    demand = demand_to_map(constants.get_demand())

    def get_score(solution: list[models.SolutionEntry]) -> float:
        evaluator = evaluation_v2.Evaluator(
            solution, demand, server_map, datacenters, selling_prices
        )
        return evaluator.get_score()

    initial_solution = reverse.get_solution(f"merged/{seed}.json")

    best_solution = initial_solution.copy()
    current_solution = initial_solution.copy()
    current_score = get_score(current_solution)
    print(f"Seed {seed} - Initial score:", current_score)
    improved = True

    while improved:
        improved = False
        for entry in current_solution:
            # Try increasing the amount
            entry.amount += 1
            new_score = get_score(current_solution)
            if new_score > current_score:
                print(f"Seed {seed} - New score:", new_score)
                current_score = new_score
                improved = True
                best_solution = current_solution.copy()
            else:
                # If increasing didn't help, try decreasing
                entry.amount -= 2  # Subtract 2 because we added 1 before
                if entry.amount < 0:
                    entry.amount = 0
                new_score = get_score(current_solution)
                if new_score > current_score:
                    print(f"Seed {seed} - New score:", new_score)
                    current_score = new_score
                    improved = True
                    best_solution = current_solution.copy()
                else:
                    # If neither helped, revert to original
                    entry.amount += 1
            if stop:
                break

        if improved:
            json.dump(
                generate.generate(best_solution, servers),
                open(f"local/{seed}.json", "w"),
            )
        if stop:
            break

    return seed, best_solution


def main():
    seeds = [3329, 4201, 8761, 2311, 2663, 4507, 6247, 2281, 4363, 5693]

    with ProcessPoolExecutor() as executor:
        future_to_seed = {executor.submit(optimize_seed, seed): seed for seed in seeds}

        try:
            for future in as_completed(future_to_seed):
                seed, best_solution = future.result()
                print(f"Optimization completed for seed {seed}")
                json.dump(
                    generate.generate(best_solution, servers),
                    open(f"local/{seed}.json", "w"),
                )
        except KeyboardInterrupt:
            print("Optimization interrupted. Saving current best solutions...")
            global stop
            stop = True
            executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
