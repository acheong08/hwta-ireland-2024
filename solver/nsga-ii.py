import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from server import ServerTracker
from problem import DailyElementWiseProblem

def optimize_sequence(demand_vectors, selling_prices, n_days, purchase_price, maintenance_cost, energy_cost):
    server_tracker = ServerTracker()
    daily_results = []

    for day in range(n_days):
        server_tracker.remove_old_servers(day)
        problem = DailyElementWiseProblem(demand_vectors[day], 
                                            selling_prices[day],
                                            server_tracker, 
                                            day, 
                                            purchase_price, 
                                            maintenance_cost,
                                            energy_cost)

        algorithm = NSGA2(
            pop_size=100,
            n_offsprings=100,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=int),
            mutation=PM(prob=1.0/problem.n_var, eta=20, vtype=int),
            eliminate_duplicates=True
        )

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 100),
                       verbose=False)

        if res.X is not None:
            best_solution = res.X
            if len(best_solution.shape) == 1:
                best_solution = best_solution.reshape(1, -1)
            result = problem._evaluate(best_solution, {})
            best_new_values = result["new_values"][0]
            best_total_values = result["total_values"][0]
            print(f"Day {day+1} - Best solution found")
        else:
            print(f"No feasible solution found for day {day+1}")
            best_new_values = np.zeros(21)
            best_total_values = server_tracker.get_current_values()

        daily_results.append(best_total_values)
        server_tracker.update_all_servers(daily_results[day-1], best_new_values, day)
        for i in range(day):
            print("Servers:", server_tracker.servers[i])
        

    return daily_results

# Test the sequence optimization
n_days = 168
demand_vectors = np.random.randint(2, 55846, size=(n_days, 21))
selling_prices = np.ones((n_days, 21))  # All ones for now, but can be modified
purchase_prices = np.array([15000, 16000, 19500, 22000, 120000, 140000, 160000] * 3)
maintenance_cost = np.array([288, 308, 375, 423, 2310, 2695, 3080] * 3)
energy_cost = np.array([400, 460, 800, 920, 3000, 3000, 3200] * 3)


results = optimize_sequence(demand_vectors, selling_prices, n_days, purchase_prices, maintenance_cost, energy_cost)

for day, result in enumerate(results):
    if day % 10 == 0:  # Print every 10th day to reduce output
        print(f"\nDay {day+1} results:")
        print("Values:", result)
        print("Group sums:")
        print("First 7:", np.sum(result[:7]))
        print("Middle 7:", np.sum(result[7:14]))
        print("Last 7:", np.sum(result[14:]))
        print("Objective value:", np.sum(np.minimum(result, demand_vectors[day]) * selling_prices[day]))

print(results)