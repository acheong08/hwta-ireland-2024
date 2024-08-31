import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from constants import get_selling_prices

from helper import mapSellingPriceToVector


class ElementWiseProblem(Problem):

    def __init__(self, constant_vectors, weight_vectors, n_days):
        # Define the search space: even integers from 2 to 55845
        self.options = np.arange(2, 55846, 2)
        
        # We have 21 variables per day
        n_var_per_day = 21
        n_var = n_var_per_day * n_days
        
        # Lower and upper bounds are indices of our options array
        xl = np.zeros(n_var, dtype=int)
        xu = np.full(n_var, len(self.options) - 1, dtype=int)

        # Store the constant vectors and weight vectors for each day
        self.constant_vectors = np.array(constant_vectors)  # Shape: (n_days, 21)
        self.weight_vectors = np.array(weight_vectors)  # Shape: (n_days, 21)
        self.n_days = n_days

        # Define the group sums
        self.group_sums = [25245, 15300, 15300]

        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=n_days * 3, xl=xl, xu=xu, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # Reshape x to (n_populations, n_days, 21)
        x_reshaped = x.reshape(x.shape[0], self.n_days, -1)
        
        # Convert indices to actual values
        values = self.options[x_reshaped.astype(int)]
        
        daily_objectives = np.zeros((x.shape[0], self.n_days))
        daily_constraints = np.zeros((x.shape[0], self.n_days, 3))
        
        for day in range(self.n_days):
            self.server_tracker.remove_old_servers(day)
            current_values = self.server_tracker.get_current_values()
            new_values = np.maximum(values[:, day, :] - current_values, 0)
            total_values = current_values + new_values
            
            # Calculate the first part of the objective function
            part1 = np.sum(np.minimum(total_values, self.constant_vectors[day]) * self.weight_vectors[day], axis=1)
            
            # Calculate the second part of the objective function
            epsilon = 1e-10
            part2 = np.sum(np.minimum(total_values, self.constant_vectors[day]) / (total_values + epsilon), axis=1)
            
            daily_objectives[:, day] = part1 * part2
            
            daily_constraints[:, day, 0] = np.abs(np.sum(total_values[:, :7], axis=1) - self.group_sums[0])
            daily_constraints[:, day, 1] = np.abs(np.sum(total_values[:, 7:14], axis=1) - self.group_sums[1])
            daily_constraints[:, day, 2] = np.abs(np.sum(total_values[:, 14:], axis=1) - self.group_sums[2])
            
            self.server_tracker.add_servers(day, new_values[0])  # Add servers for the first solution
        
        # Negate the sum of daily objectives for maximization
        out["F"] = -np.sum(daily_objectives, axis=1).reshape(-1, 1)
        
        out["G"] = daily_constraints.reshape(x.shape[0], -1)
        
        # Find and print the argmin
        feasible_mask = np.all(out["G"] <= 1e-6, axis=1)
        if np.any(feasible_mask):
            argmin = np.argmin(out["F"][feasible_mask])
            print(f"Argmin (among feasible solutions): {argmin}")
        else:
            argmin = np.argmin(out["F"])
            print(f"Argmin (no feasible solutions): {argmin}")
        
        return out
    

def main():
    # Test the _evaluate method
    n_days = 168
    constant_vectors = np.array([np.random.randint(2, 25001, size=21) for _ in range(n_days)])
    constant_vectors = constant_vectors - constant_vectors % 2  # Ensure even numbers
    constant_vectors = constant_vectors * 25000 // np.sum(constant_vectors, axis=1, keepdims=True)  # Normalize to sum to 25000

    weight_vectors = mapSellingPriceToVector(get_selling_prices())

    problem = ElementWiseProblem(constant_vectors, weight_vectors, n_days)

    # Create a sample input (indices, not actual values)
    x = np.random.randint(0, len(problem.options), size=(10, 21 * n_days))

    # Create an output dictionary
    out = {}

    # Call the _evaluate method
    result = problem._evaluate(x, out)

    print("Constant vectors shape:", constant_vectors.shape)
    print("Weight vectors shape:", weight_vectors.shape)
    print("Input x shape:", x.shape)
    print("Normalized values shape:", result["values"].shape)
    print("Output F (sum of min(x_i, c_i) * w_i over all days):", result["F"].flatten())
    print("Output G (constraint violation for each day):", result["G"])

    # Get the argmin (index of the minimum sum)
    argmin = np.argmin(result["F"])
    print("\nArgmin (index of minimum sum):", argmin)
    print("Best solution (values):")
    print(result["values"][argmin])
    print("Group sums for best solution:")
    for day in range(n_days):
        print(f"Day {day + 1}:")
        print("First 7:", np.sum(result["values"][argmin, day, :7]))
        print("Middle 7:", np.sum(result["values"][argmin, day, 7:14]))
        print("Last 7:", np.sum(result["values"][argmin, day, 14:]))
    print("Sum of min(x_i, c_i) * w_i for best solution:", result["F"][argmin][0])
    print("Constraint violations of best solution:", result["G"][argmin])

if __name__ == "__main__":
    main()



