import numpy as np
import pandas as pd
import os

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
    
def main():
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')
    selling_prices = pd.read_csv('./data/selling_prices.csv')
    solution = solution_data_preparation(solution, servers, datacenters, selling_prices)
    print(solution)

main()
