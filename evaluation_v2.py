import statistics

import numpy as np
from scipy.stats import truncweibull_min  # type: ignore[import]

import constants
from reverse import get_solution
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
        selling_prices: list[models.SellingPrices],
        verbose: bool = False,
    ) -> None:
        for action in actions:
            if self.actions.get(action.timestep) is None:
                self.actions[action.timestep] = []
            self.actions[action.timestep].append(action)
        self.demand: dict[
            int, dict[models.ServerGeneration, dict[models.Sensitivity, int]]
        ] = {}
        for d in demand:
            if d.time_step not in self.demand:
                self.demand[d.time_step] = {}
            if d.server_generation not in self.demand[d.time_step]:
                self.demand[d.time_step][d.server_generation] = {}
            for sen in models.Sensitivity:
                self.demand[d.time_step][d.server_generation][sen] = d.get_latency(sen)
        self.server_map = {server.server_generation: server for server in servers}
        self.datacenter_map = {dc.datacenter_id: dc for dc in datacenters}
        self.selling_prices: dict[
            models.ServerGeneration, dict[models.Sensitivity, int]
        ] = {}
        for sp in selling_prices:
            if sp.server_generation not in self.selling_prices:
                self.selling_prices[sp.server_generation] = {}
            if sp.latency_sensitivity not in self.selling_prices[sp.server_generation]:
                self.selling_prices[sp.server_generation][sp.latency_sensitivity] = 0
            self.selling_prices[sp.server_generation][
                sp.latency_sensitivity
            ] = sp.selling_price
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
                    operating_time = current_time - bought_time + 1
                    cost = get_maintenance_cost(
                        self.server_map[generation].average_maintenance_fee,
                        operating_time,
                        server.life_expectancy,
                    )
                    total_cost += (
                        cost
                        * self.adjust_capacity(amount, generation)
                        / self.server_map[generation].capacity
                    )
        return total_cost

    def buying_cost(self, ts: int):
        total_cost = 0
        for generation in self.operating_servers:
            for datacenter in self.operating_servers[generation]:
                for amount, bought_time in self.operating_servers[generation][
                    datacenter
                ]:
                    if bought_time == ts:
                        total_cost += (
                            amount * self.server_map[generation].purchase_price
                        )
        return total_cost

    def revenue(self, ts: int):
        total_revenue = 0
        for generation in self.operating_servers:
            for sen in models.Sensitivity:
                amount = sum(
                    (
                        amount
                        if self.datacenter_map[datacenter].latency_sensitivity == sen
                        else 0
                    )
                    for datacenter in self.operating_servers[generation]
                    for amount, _ in self.operating_servers[generation][datacenter]
                )
                total_revenue += (
                    min(
                        self.adjust_capacity(amount, generation),
                        self.get_demand(ts, generation, sen),
                    )
                    * self.selling_prices[generation][sen]
                )

        return total_revenue

    def get_demand(
        self, ts: int, generation: models.ServerGeneration, sen: models.Sensitivity
    ):
        return self.demand[ts].get(generation, {}).get(sen, 0)

    def average_utilization(self, ts: int):
        total_utilization = 0
        count = 0
        for generation in models.ServerGeneration:
            for sen in models.Sensitivity:
                count += 1
                total_capacity = sum(
                    (
                        amount * self.server_map[generation].capacity
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
                        min(total_capacity, self.get_demand(ts, generation, sen))
                        / total_capacity
                    )
        return total_utilization / count

    def normalized_lifespan(self, ts: int) -> float:
        servers = [
            # Operating time divided by life expectancy
            (ts - bought_time + 1) / self.server_map[generation].life_expectancy
            for generation in self.operating_servers
            for datacenter in self.operating_servers[generation]
            for _, bought_time in self.operating_servers[generation][datacenter]
        ]
        return statistics.mean(servers) if len(servers) > 0 else 1.0

    def get_score(self):
        total_score = 0
        for ts in range(1, 169):
            self.do_action(ts)
            self.expire_servers(ts)
            cost = self.buying_cost(ts) + self.energy_cost() + self.maintenance_cost(ts)
            revenue = self.revenue(ts)
            profit = revenue - cost
            utilization = self.average_utilization(ts)
            life_span = self.normalized_lifespan(ts)
            score = profit * utilization * life_span
            total_score += score
            if self.verbose:
                # Round to 2 decimals
                print(
                    f"{ts}: O:{round(total_score, 2)} U:{round(utilization,2)} L:{round(life_span, 2)} P:{round(profit, 2)} R:{round(revenue,2)} C:{round(cost,2)}"
                )
        return total_score


if __name__ == "__main__":
    models.SCALE = 1
    seed = 123
    np.random.seed(seed)
    solution = get_solution(f"output/{seed}.json")

    evaluator = Evaluator(
        solution,
        constants.get_demand(),
        constants.get_servers(),
        constants.get_datacenters(),
        constants.get_selling_prices(),
        verbose=True,
    )
    print(evaluator.get_score())
