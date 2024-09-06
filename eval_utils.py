# pyright: basic
import json

import numpy as np

import constants
from evaluation import evaluation_function
from evaluation_v2 import Evaluator
from reverse import get_solution
from utils import load_problem_data, load_solution


def get_score(f: str, seed: int, rerun: bool = False, fast: bool = True) -> float:
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
    return score
