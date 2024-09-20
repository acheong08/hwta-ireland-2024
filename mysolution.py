# pyright: reportAssignmentType=false, reportUnknownMemberType=false


import json

import numpy as np
import pandas as pd

from constants import get_datacenters, get_elasticity, get_selling_prices, get_servers
from evaluation import get_actual_demand  # type: ignore[import]
from solver.models import Demand, Sensitivity
from solver.sat import create_supply_map, solve_supply

seeds: list[int] = [3329, 4201, 8761, 2311, 2663, 4507, 6247, 2281, 4363, 5693]

known_solutions: set[str] = set()

count = 0
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
    solution = solve_supply(
        parsed_demand,
        get_datacenters(),
        get_selling_prices(),
        servers,
        get_elasticity(),
    )
    demand_map = create_supply_map()
    for d in parsed_demand:
        for sen in Sensitivity:
            demand_map[d.server_generation.value][sen.value][d.time_step] = (
                d.get_latency(sen)
            )
    with open(f"output/{seed}.json", "w") as f:
        json.dump(solution, f)
    with open(f"output/{seed}_demand.json", "w") as f:
        json.dump(demand_map, f)
