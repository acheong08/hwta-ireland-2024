# pyright: basic
import json
import subprocess

import numpy as np

import constants
from evaluation_v2 import Evaluator
from reverse import get_solution


def get_score(f: str, seed: int, verbose: bool = False) -> float:
    solution_scores: dict[str, float] = json.load(open("solution_scores.json", "r"))
    md5sum = subprocess.run(
        ["md5sum", f], stdout=subprocess.PIPE, text=True
    ).stdout.split()[0]
    if not verbose:
        if md5sum in solution_scores:
            return solution_scores[md5sum]
    np.random.seed(seed)
    solution = get_solution(f)
    evaluator = Evaluator(
        solution,
        constants.get_demand(),
        constants.get_servers(),
        constants.get_datacenters(),
        constants.get_selling_prices(),
    )
    score = evaluator.get_score()
    solution_scores[md5sum] = score
    json.dump(solution_scores, open("solution_scores.json", "w"))
    return score
