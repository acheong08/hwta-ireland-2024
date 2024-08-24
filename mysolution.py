# pyright: reportAssignmentType=false, reportUnknownMemberType=false


# import json

import numpy as np
import pandas as pd

from constants import get_datacenters, get_selling_prices, get_servers
from evaluation import get_actual_demand  # type: ignore[import]
from generate import generate
from seeds import known_seeds
from solver.models import Demand
from solver.sat import solve
from utils import save_solution  # type: ignore[import]

seeds: list[int] = known_seeds("training")

for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    parsed_demand: list[Demand] = []
    for i, row in get_actual_demand(pd.read_csv("./data/demand.csv")).iterrows():  # type: ignore[reportUnknownVariableType, reportArgumentType]
        parsed_demand.append(
            Demand(row.time_step, row.server_generation, row.high, row.medium, row.low).setup()  # type: ignore[reportUnknownArgumentType]
        )
    servers = get_servers()
    solution = solve(parsed_demand, get_datacenters(), get_selling_prices(), servers)
    # Save the solution for reuse
    # json.dump(
    #     [sol.to_dict() for sol in solution], open(f"./output/{seed}_solution.json", "w")
    # )
    generated = generate(solution, servers)

    save_solution(generated.solution(), f"./output/{seed}.json")
    break
