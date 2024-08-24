import numpy as np
import pandas as pd
import random

# Problem parameters
NUM_SERVERS = 4
NUM_DAYS = 168
POPULATION_SIZE = 100

def get_known(key):
    # STORE SOME CONFIGURATION VARIABLES
    if key == 'datacenter_id':
        return ['DC1', 
                'DC2', 
                'DC3', 
                'DC4']
    elif key == 'actions':
        return ['buy',
                'hold',
                'move',
                'dismiss']
    elif key == 'server_generation':
        return ['CPU.S1', 
                'CPU.S2', 
                'CPU.S3', 
                'CPU.S4', 
                'GPU.S1', 
                'GPU.S2', 
                'GPU.S3']
    elif key == 'latency_sensitivity':
        return ['high', 
                'medium', 
                'low']
    elif key == 'required_columns':
        return ['time_step', 
                'datacenter_id', 
                'server_generation', 
                'server_id',
                'action']
    elif key == 'time_steps':
        return 168
    
def solution_data_preparation(solution, servers, datacenters, selling_prices):
    # CHECK DATA FORMAT
    solution = check_data_format(solution)
    solution = check_actions(solution)
    # CHECK DATACENTERS AND SERVERS NAMES
    solution = check_datacenters_servers_generation(solution)
    # ADD PROBLEM DATA
    solution = solution.merge(servers, on='server_generation', how='left')
    solution = solution.merge(datacenters, on='datacenter_id', how='left')
    solution = solution.merge(selling_prices, 
                              on=['server_generation', 'latency_sensitivity'], 
                              how='left')
    # CHECK IF SERVERS ARE USED AT THE RIGHT RELEASE TIME
    solution = check_server_usage_by_release_time(solution)
    # DROP DUPLICATE SERVERS IDs
    solution = drop_duplicate_server_ids(solution)
    return solution.reset_index(drop=True, inplace=False)


def check_data_format(solution):
    # CHECK THAT WE HAVE ALL AND ONLY THE REQUIRED COLUMNS
    required_cols = get_known('required_columns')
    try:
        return solution[required_cols]
    except Exception:
        raise(ValueError('Please check the solution format.'))

def check_data_format(solution):
    # CHECK THAT WE HAVE ALL AND ONLY THE REQUIRED COLUMNS
    required_cols = get_known('required_columns')
    try:
        return solution[required_cols]
    except Exception:
        raise(ValueError('Please check the solution format.'))


def check_actions(solution):
    # CHECK THAT WE ARE USING ONLY ALLOWED ACTIONS
    actions = get_known('actions')
    solution = solution[solution['action'].isin(actions)]
    if not (solution[solution['time_step'] == 1]['action'] == 'buy').all():
        raise(ValueError('At time-step 1 it is only possible to use the "buy" action.'))
    return solution.reset_index(drop=True, inplace=False)


def check_datacenters_servers_generation(solution):
    # CHECK THAT DATA-CENTERS AND SERVER GENERATIONS ARE NAMED AS REQUESTED
    known_datacenters = get_known('datacenter_id')
    known_generations = get_known('server_generation')
    solution = solution[solution['datacenter_id'].isin(known_datacenters)]
    solution = solution[solution['server_generation'].isin(known_generations)]
    return solution


def check_server_usage_by_release_time(solution):
    # CHECK THAT ONLY THE SERVERS AVAILABLE FOR PURCHASE AT A CERTAIN TIME-STEP
    # ARE USED AT THAT TIME-STEP
    solution['rt_is_fine'] = solution.apply(check_release_time, axis=1)
    solution = solution[solution['rt_is_fine']]
    solution = solution.drop(columns='rt_is_fine', inplace=False)
    return solution


def check_release_time(x):
    # HELPER FUNCTION TO CHECK THE CORRECT SERVER USAGE BY TIME-STEP
    rt = eval(x['release_time'])
    ts = x['time_step']
    if ts >= min(rt) and ts <= max(rt):
        return True
    else:
        return False


def drop_duplicate_server_ids(solution):
    # DROP SERVERS THAT ARE BOUGHT MULTIPLE TIMES WITH THE SAME SERVER ID
    drop = solution[(solution['server_id'].duplicated()) & (solution['action'] == 'buy')].index
    if drop.any():
        solution = solution.drop(index=drop, inplace=False)
    return solution


def change_selling_prices_format(selling_prices):
    # ADJUST THE FORMAT OF THE SELLING PRICES DATAFRAME TO GET ALONG WITH THE
    # REST OF CODE
    selling_prices = selling_prices.pivot(index='server_generation', columns='latency_sensitivity')
    selling_prices.columns = selling_prices.columns.droplevel(0)
    return selling_prices


def get_actual_demand(demand):
    # CALCULATE THE ACTUAL DEMAND AT TIME-STEP t
    actual_demand = []
    for ls in get_known('latency_sensitivity'):
        for sg in get_known('server_generation'):
            d = demand[demand['latency_sensitivity'] == ls]
            sg_demand = d[sg].values.astype(float)
            rw = get_random_walk(sg_demand.shape[0], 0, 2)
            sg_demand += (rw * sg_demand)

            ls_sg_demand = pd.DataFrame()
            ls_sg_demand['time_step'] = d['time_step']
            ls_sg_demand['server_generation'] = sg
            ls_sg_demand['latency_sensitivity'] = ls
            ls_sg_demand['demand'] = sg_demand.astype(int)
            actual_demand.append(ls_sg_demand)

    actual_demand = pd.concat(actual_demand, axis=0, ignore_index=True)
    actual_demand = actual_demand.pivot(index=['time_step', 'server_generation'], columns='latency_sensitivity')
    actual_demand.columns = actual_demand.columns.droplevel(0)
    actual_demand = actual_demand.loc[actual_demand[get_known('latency_sensitivity')].sum(axis=1) > 0]
    actual_demand = actual_demand.reset_index(['time_step', 'server_generation'], col_level=1, inplace=False)
    return actual_demand

def get_random_walk(n, mu, sigma):
    # HELPER FUNCTION TO GET A RANDOM WALK TO CHANGE THE DEMAND PATTERN
    r = np.random.normal(mu, sigma, n)
    ts = np.empty(n)
    ts[0] = r[0]
    for i in range(1, n):
        ts[i] = ts[i - 1] + r[i]
    ts = (2 * (ts - ts.min()) / np.ptp(ts)) - 1
    return ts

class Server:
    def __init__(self, capacity, energy_cost, maintenance_cost, life_expectancy):
        self.capacity = capacity
        self.energy_cost = energy_cost
        self.maintenance_cost = maintenance_cost
        self.life_expectancy = life_expectancy

# Define your servers
servers = [
    Server(capacity=100, energy_cost=10, maintenance_cost=5, life_expectancy=200),
    Server(capacity=150, energy_cost=15, maintenance_cost=7, life_expectancy=180),
    Server(capacity=200, energy_cost=20, maintenance_cost=10, life_expectancy=160),
    Server(capacity=250, energy_cost=25, maintenance_cost=12, life_expectancy=140)
]

def create_individual():
    # Create a random allocation of servers for each day
    return [random.randint(0, NUM_SERVERS-1) for _ in range(NUM_DAYS)]

def create_population(size):
    return [create_individual() for _ in range(size)]

def calculate_fitness(individual):
    # This is where you'll implement your objective function
    # For now, we'll use a placeholder
    profit = 0
    utilization = 0
    life_expectancy = 0
    
    # Implement your calculations for P, L, and U here
    
    return profit * utilization * life_expectancy

def select_parents(population, fitness_scores):
    # Tournament selection
    tournament_size = 5
    selected = []
    for _ in range(2):
        tournament = random.sample(list(enumerate(population)), tournament_size)
        winner = max(tournament, key=lambda x: fitness_scores[x[0]])
        selected.append(winner[1])
    return selected

def crossover(parent1, parent2):
    # Single-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, NUM_SERVERS-1)
    return individual

def genetic_algorithm():
    population = create_population(POPULATION_SIZE)
    
    for generation in range(get_known('time_steps')):
        fitness_scores = [calculate_fitness(ind) for ind in population]
        
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = new_population
        
        best_fitness = max(fitness_scores)
        print(f"Generation {generation}: Best fitness = {best_fitness}")
    
    best_individual = max(population, key=calculate_fitness)
    return best_individual

# Run the genetic algorithm
best_solution = genetic_algorithm()
print("Best solution:", best_solution)