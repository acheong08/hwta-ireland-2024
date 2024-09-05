# pyright: reportUnknownMemberType=false, reportUnusedCallResult=false
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncweibull_min  # type: ignore[import]

import constants
from solver import models
from utils import demand_to_map, sp_to_map


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


class Solver:
    operating_servers: dict[models.ServerGeneration, dict[str, list[tuple[int, int]]]]
    actions: dict[int, list[models.SolutionEntry]]

    def __init__(
        self,
        actions: list[models.SolutionEntry],
        demand: (
            list[models.Demand]
            | dict[int, dict[models.ServerGeneration, dict[models.Sensitivity, int]]]
        ),
        servers: list[models.Server] | dict[models.ServerGeneration, models.Server],
        datacenters: list[models.Datacenter] | dict[str, models.Datacenter],
        selling_prices: (
            list[models.SellingPrices]
            | dict[models.ServerGeneration, dict[models.Sensitivity, int]]
        ),
        plot_generation: models.ServerGeneration | None = None,
    ) -> None:
        self.actions = {}
        self.operating_servers = {}
        for action in actions:
            if self.actions.get(action.timestep) is None:
                self.actions[action.timestep] = []
            self.actions[action.timestep].append(action)
        if isinstance(demand, list):
            self.demand = demand_to_map(demand)
        else:
            self.demand = demand
        if isinstance(servers, list):
            self.server_map = {server.server_generation: server for server in servers}
        else:
            self.server_map = servers
        if isinstance(datacenters, list):
            self.datacenter_map = {dc.datacenter_id: dc for dc in datacenters}
        else:
            self.datacenter_map = datacenters
        if isinstance(selling_prices, list):
            self.selling_prices = sp_to_map(selling_prices)
        else:
            self.selling_prices = selling_prices
        self.plot_generation = plot_generation

    def get_demand(
        self, ts: int, generation: models.ServerGeneration, sen: models.Sensitivity
    ):
        return (
            self.demand[ts].get(generation, {}).get(sen, 0)
            // self.server_map[generation].capacity
        )

    def heuristic_solve(self):
        # Sort datacenter by lowest energy cost
        cheap_datacenters = sorted(
            self.datacenter_map.values(),
            key=lambda dc: dc.cost_of_energy,
        )
        availability_by_datacenter: dict[
            int, dict[str, dict[models.ServerGeneration, int]]
        ] = {
            ts: {
                dc: {sg: 0 for sg in models.ServerGeneration}
                for dc in self.datacenter_map
            }
            for ts in range(1, 169)
        }

        for ts in range(1, 169):
            ranking: list[tuple[models.ServerGeneration, models.Sensitivity, float]] = (
                []
            )
            optimal_capacities = {
                sg: {sen: 0 for sen in models.Sensitivity}
                for sg in models.ServerGeneration
            }
            for sen in models.Sensitivity:
                # Calculate the expected profit of a single server of this type
                # Find the total capacity for this sensitivity
                slots_capacity = sum(
                    (
                        self.datacenter_map[dc].slots_capacity
                        if self.datacenter_map[dc].latency_sensitivity == sen
                        else 0
                    )
                    for dc in self.datacenter_map
                )
                for sg in models.ServerGeneration:
                    revenue = weibullshit(self.selling_prices[sg][sen])
                    energy_consumption = self.server_map[sg].energy_consumption
                    buying_cost = self.server_map[sg].purchase_price
                    maintenance_cost = sum(
                        get_maintenance_cost(
                            self.server_map[sg].average_maintenance_fee,
                            ts2,
                            self.server_map[sg].life_expectancy,
                        )
                        for ts2 in range(
                            1, min(168 - ts, self.server_map[sg].life_expectancy)
                        )
                    )
                    ranking.append(
                        (
                            sg,
                            sen,
                            revenue
                            - energy_consumption
                            - maintenance_cost
                            - buying_cost,
                        )
                    )

                    demand = self.get_demand(ts, sg, sen)
                    if demand == 0:
                        continue
                    # The optimal capacity should try to meet demand
                    meet_demand = demand
                    # But it can't exceed the datacenter capacity
                    meet_demand = (
                        min(
                            meet_demand,
                            slots_capacity // self.server_map[sg].slots_size,
                        )
                        if meet_demand > 0
                        else 0
                    )
                    c = 0
                    average_demand = 0
                    for ts2 in range(1, 169):
                        if self.get_demand(ts2, sg, sen) == 0:
                            continue
                        average_demand += self.get_demand(ts2, sg, sen)
                        c += 1
                    average_demand //= c

                    optimal_capacities[sg][sen] = min(meet_demand, average_demand)
            # Sort the ranking by profitability (most profitable first)
            ranking.sort(key=lambda x: x[2], reverse=True)
            # Reduce optimal capacities to fit within datacenter slots

            capacity_by_datacenter: dict[str, int] = {
                dc: self.datacenter_map[dc].slots_capacity for dc in self.datacenter_map
            }
            for sg, sen, _ in ranking:
                remaining_to_fill = (
                    optimal_capacities[sg][sen] * self.server_map[sg].slots_size
                )
                for dc in cheap_datacenters:
                    if dc.latency_sensitivity != sen:
                        continue
                    if remaining_to_fill == 0:
                        break
                    to_fill = min(
                        remaining_to_fill, capacity_by_datacenter[dc.datacenter_id]
                    )
                    availability_by_datacenter[ts][dc.datacenter_id][sg] += (
                        to_fill // self.server_map[sg].slots_size
                    )
                    capacity_by_datacenter[dc.datacenter_id] -= to_fill
                    remaining_to_fill -= to_fill
        return availability_by_datacenter


if __name__ == "__main__":
    models.scale = 1

    seed = 123

    np.random.seed(seed)
    demand = constants.get_demand()

    evaluator = Solver(
        [],
        demand,
        constants.get_servers(),
        constants.get_datacenters(),
        constants.get_selling_prices(),
    )
    availability = evaluator.heuristic_solve()
    gen = models.ServerGeneration.CPU_S1

    # Extract data for CPU_S1
    cpu_s1_availability = {
        sen: [
            sum(
                (
                    availability[ts][dc][gen]
                    if evaluator.datacenter_map[dc].latency_sensitivity == sen
                    else 0
                )
                for dc in evaluator.datacenter_map
            )
            for ts in range(1, 169)
        ]
        for sen in models.Sensitivity
    }
    cpu_s1_demand = {
        sen: [evaluator.get_demand(ts, gen, sen) for ts in range(1, 169)]
        for sen in models.Sensitivity
    }

    # Create the plot
    _ = plt.figure(figsize=(12, 6))

    # Plot availability for each datacenter
    for sen, avail in cpu_s1_availability.items():
        plt.plot(range(1, 169), avail, label=f"Availability {sen}")

    # Plot demand for each sensitivity
    for sen, dem in cpu_s1_demand.items():
        plt.plot(range(1, 169), dem, label=f"Demand {sen.name}", linestyle="--")

    plt.xlabel("Timestep")
    plt.ylabel("Number of Servers")
    plt.title("CPU S1 Availability and Demand")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
