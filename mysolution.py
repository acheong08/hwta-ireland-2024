# pyright: reportAssignmentType=false, reportUnknownMemberType=false
import json

import numpy as np
import pandas as pd

from constants import get_datacenters, get_elasticity, get_selling_prices, get_servers
from evaluation import get_actual_demand  # type: ignore[import]
from generate import generate_pricing, generate_solution
from solver.models import Demand, Sensitivity
from solver.sat import create_supply_map, solve_supply

seeds: list[int] = [2381, 5351, 6047, 6829, 9221, 9859, 8053, 1097, 8677, 2521]

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
    supply, solution, prices = solve_supply(
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
    with open(f"output/{seed}_supply.json", "w") as f:
        json.dump(supply, f)
    with open(f"output/{seed}.json", "w") as f:
        json.dump(
            {
                "fleet": generate_solution(solution, servers),
                "pricing_strategy": generate_pricing(prices),
            },
            f,
        )
    with open(f"output/{seed}_demand.json", "w") as f:
        json.dump(demand_map, f)
