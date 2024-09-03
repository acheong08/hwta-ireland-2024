import os
from sys import argv

import numpy as np
from scipy.stats import truncweibull_min  # type: ignore[import]

import constants
from reverse import get_solution
from solver import models
from solver.debuggy import debug_on  # type: ignore[import]


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
    operating_servers: dict[models.ServerGeneration, dict[str, list[tuple[int, int]]]]
    actions: dict[int, list[models.SolutionEntry]]
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
        self.actions = {}
        self.operating_servers = {}
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
            if a.timestep != ts:
                raise ValueError("Action timestep does not match")
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
        for i, server in enumerate(servers):
            if remaining == 0:
                break
            m = min(server[0], remaining)
            servers[i] = (server[0] - m, server[1])
            remaining -= m
        if remaining != 0:
            raise ValueError("BUG! Servers not dismissed")

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
            for _, servers in datacenters.items():
                servers[:] = [
                    (amount, bought_time)
                    for amount, bought_time in servers
                    if ts - bought_time < life_expectancy
                ]

    def adjust_capacity(self, total_servers: int, generation: models.ServerGeneration):
        return weibullshit(total_servers * self.server_map[generation].capacity)

    def energy_cost(self):
        total_cost = 0
        for generation in self.operating_servers:
            for datacenter in self.operating_servers[generation]:
                for amount, _ in self.operating_servers[generation][datacenter]:
                    total_cost += (
                        self.server_map[generation].energy_consumption
                        * self.datacenter_map[datacenter].cost_of_energy
                        * amount
                    )
        return total_cost

    @debug_on(ValueError)
    def maintenance_cost(self, current_time: int):
        total_cost = 0
        for generation, datacenters in self.operating_servers.items():
            server = self.server_map[generation]
            for dc, servers in datacenters.items():
                for amount, bought_time in servers:
                    operating_time = current_time - bought_time + 1
                    if operating_time <= 0:
                        raise ValueError("operating time should not be negative")
                    cost = get_maintenance_cost(
                        self.server_map[generation].average_maintenance_fee,
                        operating_time,
                        server.life_expectancy,
                    )
                    total_cost += (
                        cost
                        # * self.adjust_capacity(amount, generation)
                        * amount
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
        utilizations: list[float] = []
        for generation in models.ServerGeneration:
            for sen in models.Sensitivity:
                total_capacity = sum(
                    (
                        self.adjust_capacity(amount, generation)
                        if self.datacenter_map[datacenter].latency_sensitivity == sen
                        else 0
                    )
                    for datacenter in self.operating_servers.get(generation, {})
                    for amount, _ in self.operating_servers[generation][datacenter]
                )
                demand = self.get_demand(ts, generation, sen)

                if total_capacity > 0:
                    utilizations.append(min(total_capacity, demand) / total_capacity)
                elif demand > 0:
                    # Handle case where there's demand but no capacity
                    # You might want to log this situation or handle it differently
                    pass

        return sum(utilizations) / len(utilizations) if utilizations else 0

    def normalized_lifespan(self, ts: int) -> float:
        weighted_lifespans: list[float] = []
        total_amount = 0

        for generation in self.operating_servers:
            for datacenter in self.operating_servers[generation]:
                for amount, bought_time in self.operating_servers[generation][
                    datacenter
                ]:
                    if amount > 0:
                        lifespan = (ts - bought_time + 1) / self.server_map[
                            generation
                        ].life_expectancy
                        weighted_lifespans.append(lifespan * amount)
                        total_amount += amount
                    else:
                        weighted_lifespans.append(amount)  # This will be 0
                        total_amount += amount  # This will be 0

        if total_amount > 0:
            return sum(weighted_lifespans) / total_amount
        else:
            return 1.0

    def check_capacity(self):
        for datacenter in self.datacenter_map:
            utilized = sum(
                amount * self.server_map[generation].slots_size
                for generation in self.operating_servers
                for amount, _ in self.operating_servers[generation].get(datacenter, [])
            )
            if utilized > self.datacenter_map[datacenter].slots_capacity:
                raise ValueError("Server capacity exceeded")

    def get_score(self):
        try:
            total_score = 0
            for ts in range(1, 169):
                self.do_action(ts)
                self.expire_servers(ts)
                self.check_capacity()
                cost = (
                    self.buying_cost(ts)
                    + self.energy_cost()
                    + self.maintenance_cost(ts)
                )
                revenue = self.revenue(ts)
                profit = revenue - cost
                utilization = self.average_utilization(ts)
                life_span = self.normalized_lifespan(ts)
                score = profit * utilization * life_span
                total_score += score
                if self.verbose:
                    # Round to 2 decimals
                    print(
                        f"{ts}: O:{round(total_score, 2)} U:{round(utilization,2)} L:{round(life_span, 2)} P:{round(profit, 2)}"
                    )
            return total_score
        except Exception as e:
            print(e)
            return 0


if __name__ == "__main__":
    models.scale = 1

    if len(argv) != 2:
        print("Usage: evaluation_v2.py <directory>")
        exit()
    le_dir = argv[1]
    files = [f if f.endswith(".json") else "" for f in os.listdir(le_dir)]
    while "" in files:
        files.remove("")
    files.sort()
    total_score = 0
    for f in files:
        seed = 0
        if len(f.split("_")) == 2:
            seed = int(f.split("_")[0])
        else:
            seed = int(f.split(".")[0])
        np.random.seed(seed)
        solution = get_solution(f"{le_dir}/{f}")

        evaluator = Evaluator(
            solution,
            constants.get_demand(),
            constants.get_servers(),
            constants.get_datacenters(),
            constants.get_selling_prices(),
            verbose=False,
        )
        score = evaluator.get_score()
        print(f"{f}: {score}")

        total_score += score
    print(f"Average score: {total_score / len(files)}")
