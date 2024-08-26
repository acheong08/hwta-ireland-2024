# pyright: basic



from evaluation import evaluation_function

from utils import load_problem_data, load_solution



# LOAD SOLUTION

solution = load_solution("./data/solution_example.json")



# LOAD PROBLEM DATA

demand, datacenters, servers, selling_prices = load_problem_data()



# EVALUATE THE SOLUTION

score = evaluation_function(

    solution, demand, datacenters, servers, selling_prices, seed=123

)



print(f"Solution score: {score}")


