import numpy as np
from scipy.stats import truncweibull_min  # type: ignore[import]

from solver import models


def weibullshit(capacity: int):
    return int(
        capacity
        * (
            1
            - float(
                truncweibull_min.rvs(  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                    0.3, 0.05, 0.1, size=1
                ).item()  # pyright: ignore[reportAttributeAccessIssue]
            )
        )
    )


def get_maintenance_cost(
    average_maint_cost: int, operating_time: int, life_expectancy: int
) -> float:
    # CALCULATE THE CURRENT MAINTENANCE COST
    return float(
        average_maint_cost
        * (
            1
            + (
                ((1.5) * (operating_time))
                / life_expectancy
                * np.log2(((1.5) * (operating_time)) / life_expectancy)
            )
        )  # pyright: ignore[reportAny]
    )


class Evaluator:
    expiry: dict[int, dict[models.ServerGeneration, dict[str, int]]] = (
        {}
    )  # Keeps track of when servers expire
    current: dict[models.ServerGeneration, dict[str, int]] = (
        {}
    )  # Keeps track of the current servers
    actions: dict[int, list[models.SolutionEntry]] = {}

    def __init__(
        self,
        actions: list[models.SolutionEntry],
        demand: list[models.Demand],
        servers: list[models.Server],
        datacenters: list[models.Datacenter],
    ) -> None:
        for action in actions:
            if self.actions.get(action.timestep) is None:
                self.actions[action.timestep] = []
            self.actions[action.timestep].append(action)
        self.demand = {d.time_step: d for d in demand}
        self.server_map = {server.server_generation: server for server in servers}
        self.datacenter_map = {dc.datacenter_id: dc for dc in datacenters}

    def do_action(self, ts: int):
        action = self.actions.get(ts)
        if action is None:
            return
        for a in action:
            if a.action == models.Action.BUY:
                self.buy(a)
            elif a.action == models.Action.DISMISS:
                self.dismiss(a)

    def dismiss(self, a: models.SolutionEntry):
        if self.current.get(a.server_generation) is None:
            raise ValueError("No servers of this generation")
        if self.current[a.server_generation].get(a.datacenter_id) is None:
            raise ValueError("No servers of this generation in this datacenter")
        if self.current[a.server_generation][a.datacenter_id] < a.amount:
            raise ValueError("Not enough servers to dismiss")
        self.current[a.server_generation][a.datacenter_id] -= a.amount
        # Remove servers from pending expiry (closest to expiry)
        pending_removals = a.amount
        for expiry_date in sorted(self.expiry.keys()):
            if pending_removals == 0:
                break
            if self.expiry[expiry_date].get(a.server_generation) is None:
                continue
            if (
                self.expiry[expiry_date][a.server_generation].get(a.datacenter_id)
                is None
            ):
                continue
            if self.expiry[expiry_date][a.server_generation][a.datacenter_id] == 0:
                continue
            min_to_remove = min(
                pending_removals,
                self.expiry[expiry_date][a.server_generation][a.datacenter_id],
            )
            self.expiry[expiry_date][a.server_generation][
                a.datacenter_id
            ] -= min_to_remove
            pending_removals -= min_to_remove

    def buy(self, a: models.SolutionEntry):
        if self.current.get(a.server_generation) is None:
            self.current[a.server_generation] = {}
        if self.current[a.server_generation].get(a.datacenter_id) is None:
            self.current[a.server_generation][a.datacenter_id] = 0
        self.current[a.server_generation][a.datacenter_id] += a.amount
        # Add servers to pending expiry
        expiry_date = a.timestep + self.server_map[a.server_generation].life_expectancy
        if self.expiry.get(expiry_date) is None:
            self.expiry[expiry_date] = {}
        if self.expiry[expiry_date].get(a.server_generation) is None:
            self.expiry[expiry_date][a.server_generation] = {}
        if self.expiry[expiry_date][a.server_generation].get(a.datacenter_id) is None:
            self.expiry[expiry_date][a.server_generation][a.datacenter_id] = 0
        self.expiry[expiry_date][a.server_generation][a.datacenter_id] += a.amount

    def expire_servers(self, ts: int):
        if self.expiry.get(ts) is None:
            return
        for generation in self.expiry[ts]:
            for datacenter in self.expiry[ts][generation]:
                if (
                    self.current[generation][datacenter]
                    < self.expiry[ts][generation][datacenter]
                ):
                    raise ValueError("More servers expired than available")
                self.current[generation][datacenter] -= self.expiry[ts][generation][
                    datacenter
                ]
                # Remove from expiry
                self.expiry[ts][generation][datacenter] = 0

    def capacity(self, generation: models.ServerGeneration, datacenter: str):
        if self.current.get(generation) is None:
            return 0
        if self.current[generation].get(datacenter) is None:
            return 0
        return weibullshit(
            self.current[generation][datacenter] * self.server_map[generation].capacity
        )

    def energy_cost(self):
        total_cost = 0
        for generation in self.current:
            for datacenter in self.current[generation]:
                total_cost += (
                    self.server_map[generation].energy_consumption
                    * self.datacenter_map[datacenter].cost_of_energy
                    * self.current[generation][datacenter]
                )
        return total_cost

    def average_utilization(self):
        pass

    def get_score(self):
        pass
