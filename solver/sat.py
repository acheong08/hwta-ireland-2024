# pyright: reportAssignmentType=false
import itertools
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from ortools.sat.python import cp_model

from .models import (
    Action,
    Datacenter,
    Demand,
    Elasticity,
    PriceEntry,
    SellingPrices,
    Sensitivity,
    Server,
    ServerGeneration,
    SolutionEntry,
)

# t = "timestep"
# d = "datacenter"
# s = "server_generation"
# a = "actions"
# am = "amount"

INFINITY: int = 2**43

MIN_TS = 1
MAX_TS = 168
K = TypeVar("K")
V = TypeVar("V")


class NestedDefaultDict(defaultdict[K, V]):
    def __init__(self, default_factory: Callable[[], V]):
        super().__init__(lambda: default_factory())


def create_supply_map() -> (
    NestedDefaultDict[str, NestedDefaultDict[str, defaultdict[int, int]]]
):
    return NestedDefaultDict(lambda: NestedDefaultDict(lambda: defaultdict(int)))


def solve_supply(
    demands: list[Demand],
    datacenters: list[Datacenter],
    selling_prices: list[SellingPrices],
    servers: list[Server],
    elasticity: list[Elasticity],
):
    elasticity_map: dict[ServerGeneration, dict[Sensitivity, float]] = {}
    for el in elasticity:
        if elasticity_map.get(el.server_generation) is None:
            elasticity_map[el.server_generation] = {}
        elasticity_map[el.server_generation][el.latency_sensitivity] = el.elasticity
    sg_map = {server.server_generation: server for server in servers}
    dc_map = {dc.datacenter_id: dc for dc in datacenters}
    demand_map: dict[int, dict[ServerGeneration, dict[Sensitivity, int]]] = {}
    for demand in demands:
        if demand_map.get(demand.time_step) is None:
            demand_map[demand.time_step] = {}
        if demand_map[demand.time_step].get(demand.server_generation) is None:
            demand_map[demand.time_step][demand.server_generation] = {}
        for sen in Sensitivity:
            demand_map[demand.time_step][demand.server_generation][sen] = (
                demand.get_latency(sen)
            )
    sp_map: dict[ServerGeneration, dict[Sensitivity, int]] = {}
    for sp in selling_prices:
        if sp_map.get(sp.server_generation) is None:
            sp_map[sp.server_generation] = {}

        sp_map[sp.server_generation][sp.latency_sensitivity] = sp.selling_price

    cp = cp_model.CpModel()
    """
    The action model is what will be solved by SAT. It decides when to buy, sell, or move servers.
    """
    action_model = {
        timestep: {
            datacenter.datacenter_id: {
                server_generation: {
                    act: cp.new_int_var(
                        0,
                        (
                            0
                            if (
                                sg_map[server_generation].release_time[0] > timestep
                                and sg_map[server_generation].release_time[1] < timestep
                                and act == Action.BUY
                            )
                            or (timestep == 0 and act == Action.DISMISS)
                            else (
                                dc_map[datacenter.datacenter_id].slots_capacity
                                // sg_map[server_generation].slots_size
                            )
                        ),
                        f"{timestep}_{datacenter}_{server_generation}_action",
                    )
                    for act in Action
                }
                for server_generation in ServerGeneration
            }
            for datacenter in datacenters
        }
        for timestep in range(1, MAX_TS + 1)
    }

    # We calculate the total cost of buying servers by multiplying to volume to price
    buying_cost = cp.new_int_var(0, INFINITY, "cost")
    _ = cp.add(
        buying_cost
        == sum(
            action_model[t][d][s][Action.BUY] * sg_map[s].purchase_price
            for t in action_model
            for d in action_model[t]
            for s in action_model[t][d]
        )
    )
    dismissed_servers = {
        t: {
            sg: {
                dc.datacenter_id: cp.new_int_var(
                    0,
                    (dc_map[dc.datacenter_id].slots_capacity // sg_map[sg].slots_size),
                    f"{t}_{sg}_{dc}_dismissed",
                )
                for dc in datacenters
            }
            for sg in ServerGeneration
        }
        for t in action_model
    }
    dismissed_servers[0] = {
        sg: {
            dc.datacenter_id: cp.new_int_var(0, 0, f"0_{sg}_{dc}_dismissed")
            for dc in datacenters
        }
        for sg in ServerGeneration
    }
    # Now we need to calculate the total availability of each type of server at each timestep
    # based on the sum of purchase amounts minus the sum of sell amounts
    # Customers don't really care about cost of energy and stuff like that. We can deal with that later
    supply = {
        t: {
            sg: {
                dc.datacenter_id: cp.new_int_var(
                    0,
                    (dc_map[dc.datacenter_id].slots_capacity // sg_map[sg].slots_size),
                    f"{t}_{sg}_{dc}_avail",
                )
                for dc in datacenters
            }
            for sg in ServerGeneration
        }
        for t in action_model
    }
    # HACK
    supply[0] = {
        sg: {
            dc.datacenter_id: cp.new_int_var(0, 0, f"0_{sg}_{dc}_avail")
            for dc in datacenters
        }
        for sg in ServerGeneration
    }

    for ts in supply:
        if ts == 0:
            continue

        for server_generation in supply[ts]:
            for dc in supply[ts][server_generation]:
                # Find expired servers that were not dismissed
                _ = cp.add(
                    dismissed_servers[ts][server_generation][dc]
                    == dismissed_servers[ts - 1][server_generation][dc]
                    + action_model[ts][dc][server_generation][Action.DISMISS]
                )
                m = cp.new_int_var(0, INFINITY, f"{ts}_{server_generation}_{dc}_m")
                if ts - sg_map[server_generation].life_expectancy >= 1:
                    _ = cp.add_max_equality(
                        m,
                        [
                            0,
                            action_model[
                                ts - sg_map[server_generation].life_expectancy
                            ][dc][server_generation][Action.BUY]
                            - dismissed_servers[ts - 1][server_generation][dc]
                            + dismissed_servers[
                                ts - sg_map[server_generation].life_expectancy - 1
                            ][server_generation][dc],
                        ],
                    )
                else:
                    _ = cp.add(m == 0)
                # Logic: we sum buy/sells for datacenters that match the sensitivity and subtract the sells
                # We do this for all timesteps in the past
                _ = cp.add(
                    # Calculate current sum
                    supply[ts][server_generation][dc]
                    == action_model[ts][dc][server_generation][Action.BUY]
                    # Take the previous timestep
                    + supply[ts - 1][server_generation][dc]
                    # Subtract dismissed servers
                    - action_model[ts][dc][server_generation][Action.DISMISS]
                    # Subtract the expired servers based on life expectancy
                    - (m if ts > sg_map[server_generation].life_expectancy else 0)
                )

    energy_cost = cp.new_int_var(0, INFINITY, "energy_cost")
    _ = cp.add(
        energy_cost
        == sum(
            (
                supply[ts][sg][dc]
                * sg_map[sg].energy_consumption
                * dc_map[dc].cost_of_energy
            )
            for ts in supply
            for sg in supply[ts]
            for dc in supply[ts][sg]
        )
    )

    maintenance_cost = 0

    for ts in supply:
        if ts == 0:
            continue
        for dc in datacenters:
            # Ensure we don't run out of slots on datacenters
            _ = cp.add(
                sum(
                    supply[ts][sg][dc.datacenter_id] * sg_map[sg].slots_size
                    for sg in supply[ts]
                )
                < dc_map[dc.datacenter_id].slots_capacity
            )

    # Calculate server utilization
    # This is the ratio of demand to availability for server type (sensitivity + server generation)
    revenues = {
        ts: {
            sg: {
                sen: cp.new_int_var(0, INFINITY, f"{ts}_{sg}_{sen}_rev")
                for sen in Sensitivity
            }
            for sg in ServerGeneration
        }
        for ts in supply
    }
    revenues[0] = {
        sg: {sen: cp.new_int_var(0, 0, f"0_{sg}_{sen}_util") for sen in Sensitivity}
        for sg in ServerGeneration
    }
    for ts in revenues:
        if ts == 0:
            continue

        for sg in revenues[ts]:
            maintenance_cost += sum(
                supply[ts][sg][dc.datacenter_id] * sg_map[sg].average_maintenance_fee
                for dc in datacenters
            )
            for sen in revenues[ts][sg]:
                total_availability = sum(
                    (
                        supply[ts][sg][dc.datacenter_id] * sg_map[sg].capacity
                        if dc.latency_sensitivity == sen
                        else 0
                    )
                    for dc in datacenters
                )
                if ts < sg_map[sg].release_time[0]:
                    _ = cp.add(revenues[ts][sg][sen] == 0)
                    continue

                # Get amount of demand that can be satisfied
                m = cp.new_int_var(0, INFINITY, f"{ts}_{sg}_{sen}_m")
                _ = cp.add_min_equality(
                    m,
                    [
                        demand_map[ts].get(sg, {sen: 0})[sen],
                        total_availability,
                    ],  # Each server has *capacity* number of cpu/gpu that satisfies demand
                )
                _ = cp.add_multiplication_equality(
                    revenues[ts][sg][sen], [m, sp_map[sg][sen]]
                )

    total_cost = cp.new_int_var(0, INFINITY, "total_cost")
    _ = cp.add(total_cost == buying_cost + energy_cost + maintenance_cost)
    total_revenue = sum(
        revenues[ts][sg][sen]
        for ts in revenues
        for sg in revenues[ts]
        for sen in revenues[ts][sg]
    )
    cp.maximize(total_revenue - total_cost)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60 * 30
    status = solver.solve(cp)
    if (
        status == cp_model.OPTIMAL  # type: ignore[reportUnnecessaryComparison]
        or status == cp_model.FEASIBLE  # type: ignore[reportUnnecessaryComparison]
    ):
        print("Time:", solver.UserTime())
        print("Status:", solver.status_name(status))
        supply_map = create_supply_map()
        for ts, sg, dc in itertools.product(
            range(MIN_TS, MAX_TS + 1), ServerGeneration, datacenters
        ):
            supply_map[sg.value][dc.latency_sensitivity.value][ts] += (
                solver.value(supply[ts][sg][dc.datacenter_id]) * sg_map[sg].capacity
            )
        solution: list[SolutionEntry] = []
        for ts in action_model:
            for dc in action_model[ts]:
                for sg in action_model[ts][dc]:
                    for act in action_model[ts][dc][sg]:
                        solution.append(
                            SolutionEntry(
                                ts,
                                dc,
                                sg,
                                act,
                                solver.value(action_model[ts][dc][sg][act]),
                            )
                        )
        prices: list[PriceEntry] = []
        for sen in Sensitivity:
            for sg in ServerGeneration:
                prices.append(PriceEntry(1, sg, sen, sp_map[sg][sen]))
        return supply_map, solution, prices
    else:
        print(solver.status_name(status))
        print(solver.solution_info())
        print(solver.response_stats())
        raise Exception("No solution found")
