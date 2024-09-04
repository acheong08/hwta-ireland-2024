# pyright: basic
import json
import subprocess

import numpy as np

import constants
from evaluation import evaluation_function
from evaluation_v2 import Evaluator
from reverse import get_solution
from solver import models
from utils import load_problem_data, load_solution


def get_score(f: str, seed: int, rerun: bool = False, fast: bool = True) -> float:
    solution_scores: dict[str, float] = json.load(open("solution_scores.json", "r"))
    md5sum = subprocess.run(
        ["md5sum", f], stdout=subprocess.PIPE, text=True
    ).stdout.split()[0]
    if not rerun:
        if md5sum in solution_scores:
            return solution_scores[md5sum]
    np.random.seed(seed)
    score = 0
    if fast:
        solution = get_solution(f)
        evaluator = Evaluator(
            solution,
            constants.get_demand(),
            constants.get_servers(),
            constants.get_datacenters(),
            constants.get_selling_prices(),
        )
        score = evaluator.get_score()
    else:
        solution = load_solution(f)
        demand, datacenters, servers, selling_prices = load_problem_data()

        score: float = evaluation_function(  # type: ignore[reportAssignmentType]
            solution, demand, datacenters, servers, selling_prices, seed=seed
        )
    solution_scores[md5sum] = score
    json.dump(solution_scores, open("solution_scores.json", "w"))
    return score


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
