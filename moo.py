# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportAny=false
import itertools
from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.moo.nsga2 import NSGA2, PM, SBX
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize  # pyright: ignore[reportUnknownVariableType]

import solver.models as models
from constants import get_datacenters, get_demand, get_selling_prices, get_servers
from evaluation_v2 import Evaluator
from utils import demand_to_map, sp_to_map

models.scale = 1

# REMEMBER TO OFFSET BY 1
MIN_TS = 0
MAX_TS = 167

SERVERS = get_servers()
DATACENTERS = get_datacenters()
SELLING_PRICES = get_selling_prices()

SERVER_MAP = {server.server_generation: server for server in SERVERS}
DATACENTER_MAP = {dc.datacenter_id: dc for dc in DATACENTERS}
SELLING_PRICES_MAP = sp_to_map(SELLING_PRICES)


N_VAR = MAX_TS * len(SERVERS) * len(models.ServerGeneration) * len(models.Action)


class MyProblem(Problem):

    def __init__(
        self,
        demand: list[models.Demand],
    ):
        self.demand = demand_to_map(demand)

        upper_bounds = np.ones(N_VAR) * 10
        for n, comb in enumerate(
            itertools.product(
                range(MIN_TS, MAX_TS),
                DATACENTER_MAP,
                models.ServerGeneration,
                models.Action,
            )
        ):
            if comb[0] == 0 and comb[3] != models.Action.BUY:
                upper_bounds[n] = 0
                continue
            # Check release time
            if (
                comb[0] < SERVER_MAP[comb[2]].release_time[0]
                or comb[0] > SERVER_MAP[comb[2]].release_time[1]
            ):
                upper_bounds[n] = 0
                continue
            # upper_bounds[n] = (
            #     DATACENTER_MAP[comb[1]].slots_capacity
            #     // SERVER_MAP[comb[2]].slots_size
            #     // 4
            # )

        super().__init__(
            n_var=N_VAR,
            n_obj=1,
            n_ieq_constr=1,
            xl=np.zeros(N_VAR),
            xu=upper_bounds,
            vtype=int,
        )

    def decode_actions(self, x: NDArray[np.int32]) -> list[models.SolutionEntry]:
        actions: list[models.SolutionEntry] = []
        for n, comb in enumerate(
            itertools.product(
                range(MIN_TS, MAX_TS),
                DATACENTER_MAP,
                models.ServerGeneration,
                models.Action,
            )
        ):
            if x[n] == 0.0:
                continue
            amount = int(x[n])
            if amount == 0:
                continue
            actions.append(
                models.SolutionEntry(
                    comb[0] + 1,
                    comb[1],
                    comb[2],
                    comb[3],
                    amount,
                )
            )
        return actions

    @override
    def _evaluate(self, x: NDArray[np.float64], out: dict[str, Any]):
        n_individuals = int(x.shape[0])
        f = np.zeros((n_individuals, 1))  # Shape: (n_individuals, n_objectives)
        # Constraints
        g = np.zeros((n_individuals, 1))

        for i in range(n_individuals):
            actions = self.decode_actions(x[i])
            evaluator = Evaluator(
                actions, self.demand, SERVER_MAP, DATACENTER_MAP, SELLING_PRICES_MAP
            )
            score = evaluator.get_score()
            f[i, 0] = -score  # Negative because we're minimizing

            # Set constraint: g <= 0 is feasible, g > 0 is infeasible
            g[i, 0] = 1 if score == -1 else -1
            # if score > 1:
            #     json.dump(
            #         generate(actions, SERVERS),
            #         open("output/2281.json", "w"),
            #         indent=" ",
            #     )

        out["F"] = f
        out["G"] = g


if __name__ == "__main__":
    algorithm = NSGA2(
        pop_size=100,
        eliminate_duplicates=True,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int),
        mutation=PM(prob=1.0 / N_VAR, eta=20, vtype=int),
    )
    np.random.seed(2281)
    demand = get_demand()
    problem = MyProblem(demand)
    res: None | Any = minimize(problem, algorithm, termination=("n_gen", 100), verbose=True)  # type: ignore[reportUnknownVariableType]
    if res is None or res.X is None:
        print("No solution found")
        exit()
    best_solution = problem.decode_actions(res.X[0])  # type: ignore[reportUnknownArgumentType]
    best_score = Evaluator(
        best_solution, demand, SERVER_MAP, DATACENTER_MAP, SELLING_PRICES_MAP
    ).get_score()
    print(f"Best score: {best_score}")
