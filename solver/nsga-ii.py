import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

class ServerTracker:
    def __init__(self, n_slots):
        self.servers = [[] for _ in range(n_slots)]
    
    def add_servers(self, day, values):
        for i, value in enumerate(values):
            if value > 0:
                self.servers[i].append((day, value))
    
    def remove_old_servers(self, current_day):
        for slot in self.servers:
            slot[:] = [(day, value) for day, value in slot if current_day - day < 96]
    
    def get_current_values(self):
        return np.array([sum(value for _, value in slot) for slot in self.servers])

class DailyElementWiseProblem(Problem):
    def __init__(self, constant_vector, weight_vector, server_tracker, current_day):
        self.options = np.arange(0, 55846, 2)
        n_var = 21
        xl = np.zeros(n_var, dtype=int)
        xu = np.full(n_var, len(self.options) - 1, dtype=int)

        self.constant_vector = np.array(constant_vector)
        self.weight_vector = np.array(weight_vector)
        self.server_tracker = server_tracker
        self.current_day = current_day
        self.group_sums = [25245, 15300, 15300]

        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=3, xl=xl, xu=xu, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        values = self.options[x.astype(int)]
        current_values = self.server_tracker.get_current_values()
        
        # Ensure values and current_values have compatible shapes
        if len(values.shape) == 2:
            current_values = current_values.reshape(1, -1)
        
        new_values = np.maximum(values - current_values, 0)

        total_values = current_values + new_values

        # Calculate the objective function
        part1 = np.sum(np.minimum(total_values, self.constant_vector) * self.weight_vector, axis=1)
        epsilon = 1e-10
        part2 = np.sum(np.minimum(total_values, self.constant_vector) / (total_values + epsilon), axis=1)
        out["F"] = -(part1 * part2).reshape(-1, 1)

        # Inequality constraints: group sums should be less than or equal to the specified values
        out["G"] = np.column_stack([
            np.sum(total_values[:, :7], axis=1) - self.group_sums[0],
            np.sum(total_values[:, 7:14], axis=1) - self.group_sums[1],
            np.sum(total_values[:, 14:], axis=1) - self.group_sums[2]
        ])

        out["new_values"] = new_values
        out["total_values"] = total_values

        return out

def optimize_sequence(constant_vectors, weight_vectors, n_days):
    server_tracker = ServerTracker(21)
    daily_results = []

    for day in range(n_days):
        server_tracker.remove_old_servers(day)
        problem = DailyElementWiseProblem(constant_vectors[day], weight_vectors[day], server_tracker, day)

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

        server_tracker.add_servers(day, best_new_values)
        daily_results.append(best_total_values)

    return daily_results

# Test the sequence optimization
n_days = 168
constant_vectors = np.random.randint(2, 55846, size=(n_days, 21))
constant_vectors = constant_vectors - constant_vectors % 2  # Ensure even numbers
weight_vectors = np.ones((n_days, 21))  # All ones for now, but can be modified

results = optimize_sequence(constant_vectors, weight_vectors, n_days)

for day, result in enumerate(results):
    if day % 10 == 0:  # Print every 10th day to reduce output
        print(f"\nDay {day+1} results:")
        print("Values:", result)
        print("Group sums:")
        print("First 7:", np.sum(result[:7]))
        print("Middle 7:", np.sum(result[7:14]))
        print("Last 7:", np.sum(result[14:]))
        print("Objective value:", np.sum(np.minimum(result, constant_vectors[day]) * weight_vectors[day]))