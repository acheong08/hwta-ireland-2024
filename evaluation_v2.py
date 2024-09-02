import statistics

import numpy as np
from scipy.stats import truncweibull_min  # type: ignore[import]

from solver import models


def weibullshit(capacity: int):
    return int(
        capacity
        * (
            1
            - float(
                truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()  # type: ignore[reportUnknownArgumentType]
            )
        )
    )


def get_maintenance_cost(
    average_maint_cost: int, operating_time: int, life_expectancy: int
) -> float:
    return float(
        average_maint_cost  # type: ignore[reportAny]
        * (
            1
            + (
                ((1.5) * (operating_time))
                / life_expectancy
                * np.log2(((1.5) * (operating_time)) / life_expectancy)
            )
        )
    )


class Evaluator:
    operating_servers: dict[
        models.ServerGeneration, dict[str, list[tuple[int, int]]]
    ] = {}
    actions: dict[int, list[models.SolutionEntry]] = {}
    verbose: bool = False

    def __init__(
        self,
        actions: list[models.SolutionEntry],
        demand: list[models.Demand],
        servers: list[models.Server],
        datacenters: list[models.Datacenter],
        verbose: bool = False,
    ) -> None:
        for action in actions:
            if self.actions.get(action.timestep) is None:
                self.actions[action.timestep] = []
            self.actions[action.timestep].append(action)
        self.demand = {d.time_step: d for d in demand}
        self.server_map = {server.server_generation: server for server in servers}
        self.datacenter_map = {dc.datacenter_id: dc for dc in datacenters}
        self.verbose = verbose

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
        if self.operating_servers.get(a.server_generation) is None:
            raise ValueError("No servers of this generation")
        if self.operating_servers[a.server_generation].get(a.datacenter_id) is None:
            raise ValueError("No servers of this generation in this datacenter")

        servers = self.operating_servers[a.server_generation][a.datacenter_id]
        total_servers = sum(amount for amount, _ in servers)
        if total_servers < a.amount:
            raise ValueError("Not enough servers to dismiss")

        remaining: int = a.amount
        while remaining > 0 and servers:
            amount, bought_time = servers[0]  # type: ignore[reportUnknownVariableType]
            if amount <= remaining:
                _ = servers.pop(0)
                remaining -= amount  # type: ignore[reportUnknownVariableType]
            else:
                servers[0] = (amount - remaining, bought_time)
                remaining = 0

    def buy(self, a: models.SolutionEntry):
        if self.operating_servers.get(a.server_generation) is None:
            self.operating_servers[a.server_generation] = {}
        if self.operating_servers[a.server_generation].get(a.datacenter_id) is None:
            self.operating_servers[a.server_generation][a.datacenter_id] = []

        self.operating_servers[a.server_generation][a.datacenter_id].append(
            (a.amount, a.timestep)
        )

    def expire_servers(self, ts: int):
        for generation, datacenters in self.operating_servers.items():
            life_expectancy = self.server_map[generation].life_expectancy
            for datacenter, servers in datacenters.items():
                servers[:] = [
                    (amount, bought_time)
                    for amount, bought_time in servers
                    if ts - bought_time < life_expectancy
                ]

    def adjust_capacity(self, total_servers: int, generation: models.ServerGeneration):
        return weibullshit(total_servers * self.server_map[generation].capacity)

    def energy_cost(self):
        total_cost = 0
        for generation, datacenters in self.operating_servers.items():
            for datacenter, servers in datacenters.items():
                total_servers = sum(amount for amount, _ in servers)
                total_cost += (
                    self.server_map[generation].energy_consumption
                    * self.datacenter_map[datacenter].cost_of_energy
                    * total_servers
                )
        return total_cost

    def maintenance_cost(self, current_time: int):
        total_cost = 0
        for generation, datacenters in self.operating_servers.items():
            server = self.server_map[generation]
            for _, servers in datacenters.items():
                for amount, bought_time in servers:
                    operating_time = current_time - bought_time
                    cost = get_maintenance_cost(
                        self.server_map[generation].average_maintenance_fee,
                        operating_time,
                        server.life_expectancy,
                    )
                    total_cost += cost * self.adjust_capacity(amount, generation)
        return total_cost

    def buying_cost(self):
        total_cost = 0
        for generation in self.operating_servers:
            for datacenter in self.operating_servers[generation]:
                for amount, _ in self.operating_servers[generation][datacenter]:
                    total_cost += amount * self.server_map[generation].purchase_price
        return total_cost

    def revenue(self, ts: int):
        total_revenue = 0
        for generation in self.operating_servers:
            for datacenter in self.operating_servers[generation]:
                for amount, _ in self.operating_servers[generation][datacenter]:
                    total_revenue += min(
                        self.demand[ts].get_latency(
                            self.datacenter_map[datacenter].latency_sensitivity
                        ),
                        self.adjust_capacity(amount, generation),
                    )
        return total_revenue

    def average_utilization(self, ts: int):
        total_utilization = 0
        count = 0
        for generation in models.ServerGeneration:
            for sen in models.Sensitivity:
                count += 1
                total_capacity = sum(
                    (
                        self.adjust_capacity(amount, generation)
                        if self.datacenter_map[datacenter].latency_sensitivity == sen
                        else 0
                    )
                    for datacenter in self.operating_servers.get(generation, {})
                    for amount, _ in self.operating_servers[generation][datacenter]
                )
                if total_capacity == 0:
                    total_utilization += 1
                else:
                    total_utilization += (
                        min(total_capacity, self.demand[ts].get_latency(sen))
                        / total_capacity
                    )
        return total_utilization / count

    def normalized_lifespan(self, ts: int):
        return statistics.mean(
            # Operating time divided by life expectancy
            (ts - bought_time) / self.server_map[generation].life_expectancy
            for generation in self.operating_servers
            for datacenter in self.operating_servers[generation]
            for _, bought_time in self.operating_servers[generation][datacenter]
        )

    def get_score(self):
        total_score = 0
        for ts in range(1, 169):
            self.do_action(ts)
            self.expire_servers(ts)
            cost = self.buying_cost() + self.energy_cost() + self.maintenance_cost(ts)
            revenue = self.revenue(ts)
            profit = revenue - cost
            utilization = self.average_utilization(ts)
            life_span = self.normalized_lifespan(ts)
            score = profit * utilization * life_span
            total_score += score
            if self.verbose:
                print(f"{ts}: P:{profit}, U:{utilization}, L:{life_span}, S:{score}")
        return total_score
