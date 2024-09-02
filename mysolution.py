# pyright: reportAssignmentType=false, reportUnknownMemberType=false


import hashlib
import json

import numpy as np
import pandas as pd

from constants import get_datacenters, get_selling_prices, get_servers
from evaluation import get_actual_demand  # type: ignore[import]
from generate import generate
from seeds import known_seeds
from solver.models import Demand
from solver.sat import solve

seeds: list[int] = known_seeds("training")

known_solutions: set[str] = set()

count = 0
while True:
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
        solution = solve(
            parsed_demand, get_datacenters(), get_selling_prices(), servers
        )

        generated = generate(solution, servers)

        s = json.dumps(generated)
        solution_hash = hashlib.md5(s.encode("ascii")).hexdigest()
        if solution_hash in known_solutions:
            print(f"Found duplicate for {seed}")
            continue
        known_solutions.add(solution_hash)

        with open(f"output/{seed}_{count}.json", "w") as f:
            _ = f.write(s)
    count += 1
    break
