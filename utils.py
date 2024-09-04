import json
from os.path import abspath, join

import pandas as pd

from solver import models


def load_json(path):
    return json.load(open(path, encoding="utf-8"))


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False, indent=4)


def load_solution(path):
    # Loads a solution from a json file to a pandas DataFrame.
    return pd.read_json(path)


def save_solution(solution, path):
    # Saves a solution into a json file.
    if isinstance(solution, pd.DataFrame):
        solution = solution.to_dict("records")
    return save_json(path, solution)


def load_problem_data(path=None):
    if path is None:
        path = "./data/"

    # LOAD DEMAND
    p = abspath(join(path, "demand.csv"))
    demand = pd.read_csv(p)

    # LOAD DATACENTERS DATA
    p = abspath(join(path, "datacenters.csv"))
    datacenters = pd.read_csv(p)

    # LOAD SERVERS DATA
    p = abspath(join(path, "servers.csv"))
    servers = pd.read_csv(p)

    # LOAD SELLING PRICES DATA
    p = abspath(join(path, "selling_prices.csv"))
    selling_prices = pd.read_csv(p)
    return demand, datacenters, servers, selling_prices


def sp_to_map(selling_prices: list[models.SellingPrices]):
    sp_map: dict[models.ServerGeneration, dict[models.Sensitivity, int]] = {}
    for sp in selling_prices:
        if sp.server_generation not in sp_map:
            sp_map[sp.server_generation] = {}
        if sp.latency_sensitivity not in sp_map[sp.server_generation]:
            sp_map[sp.server_generation][sp.latency_sensitivity] = 0
        sp_map[sp.server_generation][sp.latency_sensitivity] = sp.selling_price
    return sp_map


def demand_to_map(demand: list[models.Demand]):
    demand_map: dict[
        int, dict[models.ServerGeneration, dict[models.Sensitivity, int]]
    ] = {}
    for d in demand:
        if d.time_step not in demand_map:
            demand_map[d.time_step] = {}
        if d.server_generation not in demand_map[d.time_step]:
            demand_map[d.time_step][d.server_generation] = {}
        for sen in models.Sensitivity:
            demand_map[d.time_step][d.server_generation][sen] = d.get_latency(sen)
    return demand_map


if __name__ == "__main__":

    # Load solution
    path = "./data/solution_example.json"

    solution = load_solution(path)

    print(solution)

    # Save solution
    # path = './data/solution_example_test.json'
    # save_solution(solution, path)
