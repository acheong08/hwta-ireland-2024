# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportAny=false
import itertools
from typing import override

import numpy as np
from pymoo.core.problem import Problem

from eval_utils import demand_to_map, sp_to_map

from ..constants import get_datacenters, get_selling_prices, get_servers
from ..evaluation_v2 import Evaluator
from .models import Action, Demand, ServerGeneration, SolutionEntry

# REMEMBER TO OFFSET BY 1
MIN_TS = 0
MAX_TS = 167

SERVERS = get_servers()
DATACENTERS = get_datacenters()
SELLING_PRICES = get_selling_prices()

SERVER_MAP = {server.server_generation: server for server in SERVERS}
DATACENTER_MAP = {dc.datacenter_id: dc for dc in DATACENTERS}
SELLING_PRICES_MAP = sp_to_map(SELLING_PRICES)


class MyProblem(Problem):

    def __init__(
        self,
        demand: list[Demand],
    ):
        self.demand = demand_to_map(demand)
        n_var = MAX_TS * len(SERVERS) * len(ServerGeneration) * len(Action)

        upper_bounds = np.array([1] * n_var)
        for n, comb in enumerate(
            itertools.product(
                range(MIN_TS, MAX_TS), DATACENTER_MAP, ServerGeneration, Action
            )
        ):
            if comb[0] == 0 and comb[3] != Action.BUY:
                upper_bounds[n] = 0
                continue
            upper_bounds[n] = (
                SERVER_MAP[comb[2]].slots_size // DATACENTER_MAP[comb[1]].slots_capacity
            )

        super().__init__(
            n_var,
            xl=np.zeros(n_var),
            xu=upper_bounds,
        )

    def _decode_actions(self, x: np.ndarray):
        actions: list[SolutionEntry] = []
        for n, comb in enumerate(
            itertools.product(
                range(MIN_TS, MAX_TS), DATACENTER_MAP, ServerGeneration, Action
            )
        ):
            if x[n] == 0:
                continue
            actions.append(
                SolutionEntry(
                    comb[0],
                    comb[1],
                    comb[2],
                    comb[3],
                    int(x[n]),
                )
            )
        return actions

    @override
    def _evaluate(self, x: np.ndarray, out: dict[str, np.ndarray]):
        actions = self._decode_actions(x)
        evaluator = Evaluator(
            actions, self.demand, SERVER_MAP, DATACENTER_MAP, SELLING_PRICES_MAP
        )
        out["F"] = np.array([-evaluator.get_score()])
