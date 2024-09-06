# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportAny=false, reportUnknownArgumentType=false
import itertools
from typing import Any, override

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import Problem
from pymoo.optimize import minimize  # pyright: ignore[reportUnknownVariableType]
from pymoo.termination import get_termination  # type: ignore[reportUnknownVariableType]

import solver.models as models
from constants import get_datacenters, get_demand, get_selling_prices, get_servers
from evaluation_v2 import Evaluator
from reverse import get_solution
from utils import demand_to_map, sp_to_map

models.scale = 1

# REMEMBER TO OFFSET BY 1
MIN_TS = 1
MAX_TS = 168

SERVERS = get_servers()
DATACENTERS = get_datacenters()
SELLING_PRICES = get_selling_prices()

SERVER_MAP = {server.server_generation: server for server in SERVERS}
DATACENTER_MAP = {dc.datacenter_id: dc for dc in DATACENTERS}
SELLING_PRICES_MAP = sp_to_map(SELLING_PRICES)


N_VAR = (MAX_TS - 1) * len(SERVERS) * len(models.ServerGeneration) * len(models.Action)


def demand_for(
    demand_map: dict[int, dict[models.ServerGeneration, dict[models.Sensitivity, int]]],
    ts: int,
    sg: models.ServerGeneration,
    sen: models.Sensitivity,
):
    return demand_map[ts].get(sg, {}).get(sen, 0)


class MyProblem(Problem):

    def __init__(
        self,
        demand: list[models.Demand],
    ):
        self.demand = demand_to_map(demand)

        super().__init__(
            n_var=N_VAR,
            n_obj=1,
            n_ieq_constr=1,
            xl=np.zeros(N_VAR),
            vtype=int,
        )

    def decode_actions(self, x: NDArray[np.int64]) -> list[models.SolutionEntry]:
        actions: list[models.SolutionEntry] = []
        for n, comb in enumerate(
            itertools.product(
                range(MIN_TS, MAX_TS + 1),
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

    def evaluate_individual(self, x: NDArray[np.int64]) -> tuple[float, float]:
        actions = self.decode_actions(x)
        evaluator = Evaluator(
            actions, self.demand, SERVER_MAP, DATACENTER_MAP, SELLING_PRICES_MAP
        )
        valid_solution = evaluator.quick_validate()
        if valid_solution:
            score = evaluator.get_score()
            return -score, -1  # Negative score because we're minimizing, -1 for g
        else:
            return 0, 1  # 0 for f, 1 for g (constraint violation)

    @override
    def _evaluate(self, x: NDArray[np.int64], out: dict[str, Any]):
        # n_individuals = int(x.shape[0])
        # print(n_individuals)
        # with ProcessPoolExecutor() as executor:
        #     results: list[tuple[float, float]] = list(
        #         executor.map(self.evaluate_individual, x)
        #     )
        #
        # f = np.array([r[0] for r in results]).reshape(n_individuals, 1)
        # g = np.array([r[1] for r in results]).reshape(n_individuals, 1)
        f, g = self.evaluate_individual(x[0])

        out["F"] = np.array(f)
        out["G"] = np.array(g)


def actions_to_np(actions: list[models.SolutionEntry]) -> NDArray[np.int64]:
    action_map: dict[
        int, dict[models.ServerGeneration, dict[str, dict[models.Action, int]]]
    ] = {}
    for action in actions:
        if action.timestep not in action_map:
            action_map[action.timestep] = {}
        if action.server_generation not in action_map[action.timestep]:
            action_map[action.timestep][action.server_generation] = {}
        if (
            action.datacenter_id
            not in action_map[action.timestep][action.server_generation]
        ):
            action_map[action.timestep][action.server_generation][
                action.datacenter_id
            ] = {}
        action_map[action.timestep][action.server_generation][action.datacenter_id][
            action.action
        ] = action.amount
    out = np.zeros(N_VAR, dtype=int)
    for n, comb in enumerate(
        itertools.product(
            range(MIN_TS, MAX_TS + 1),
            DATACENTER_MAP,
            models.ServerGeneration,
            models.Action,
        )
    ):
        if comb[0] not in action_map:
            continue
        if comb[2] not in action_map[comb[0]]:
            continue
        if comb[1] not in action_map[comb[0]][comb[2]]:
            continue
        if comb[3] not in action_map[comb[0]][comb[2]][comb[1]]:
            continue
        out[n] = action_map[comb[0]][comb[2]][comb[1]][comb[3]]
    return out


SEED = 2281

if __name__ == "__main__":

    initial_solution = actions_to_np(get_solution(f"merged/{SEED}.json"))
    algorithm = PatternSearch(x0=initial_solution)
    np.random.seed(2281)
    demand = get_demand()
    problem = MyProblem(demand)
    termination = get_termination("time", 1)
    res: None | Any = minimize(problem, algorithm, termination, verbose=True)  # type: ignore[reportUnknownVariableType]
    if res is None or res.X is None:
        print("No solution found")
        exit()

    best_solution = problem.decode_actions(res.X)
    best_score = Evaluator(
        best_solution, demand, SERVER_MAP, DATACENTER_MAP, SELLING_PRICES_MAP
    ).get_score()
    print(f"Best score: {best_score}")
