# pyright: reportAssignmentType=false


from ortools.sat.python import cp_model

from .models import (
    Action,
    Datacenter,
    Demand,
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


def solve(
    demands: list[Demand],
    datacenters: list[Datacenter],
    selling_prices: list[SellingPrices],
    servers: list[Server],
) -> list[SolutionEntry]:

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
                    action: cp.new_int_var(
                        0,
                        (
                            (
                                dc_map[datacenter.datacenter_id].slots_capacity
                                // sg_map[server_generation].slots_size
                            )
                            if sg_map[server_generation].release_time[0] <= timestep
                            and sg_map[server_generation].release_time[1] >= timestep
                            else 0
                        ),
                        f"{timestep}_{datacenter}_{server_generation}_{action}_action",
                    )
                    for action in Action
                }
                for server_generation in ServerGeneration
            }
            for datacenter in datacenters
        }
        for timestep in range(1, max(demand.time_step for demand in demands) + 1)
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

    # Only one action (or less) can be taken per datacenter per timestep
    for ts in action_model:
        if ts == 0:
            continue

        for dc in action_model[ts]:
            for server_gen in action_model[ts][dc]:
                for action in Action:
                    cur_mod = cp.new_int_var(0, 1, f"{ts}_{dc}_{action}")
                    _ = cp.add_modulo_equality(
                        cur_mod, action_model[ts][dc][server_gen][action], 2
                    )
                    for other_action in Action:
                        if other_action != action:
                            ot_mod = cp.new_int_var(
                                0, 1, f"{ts}_{dc}_{action}_{other_action}"
                            )
                            _ = cp.add_modulo_equality(
                                ot_mod,
                                action_model[ts][dc][server_gen][other_action],
                                2,
                            )
                            _ = cp.add(cur_mod == 0).only_enforce_if(
                                ot_mod.is_equal_to(1)
                            )

    # Now we need to calculate the total availability of each type of server at each timestep
    # based on the sum of purchase amounts minus the sum of sell amounts
    # Customers don't really care about cost of energy and stuff like that. We can deal with that later
    availability = {
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
    availability[0] = {
        sg: {
            dc.datacenter_id: cp.new_int_var(0, 0, f"0_{sg}_{dc}_avail")
            for dc in datacenters
        }
        for sg in ServerGeneration
    }

    for ts in availability:
        if ts == 0:
            continue

        for server_generation in availability[ts]:
            for dc in availability[ts][server_generation]:

                # Logic: we sum buy/sells for datacenters that match the sensitivity and subtract the sells
                # We do this for all timesteps in the past
                _ = cp.add(
                    # Calculate current sum
                    availability[ts][server_generation][dc]
                    == (action_model[ts][dc][server_generation][Action.BUY])
                    # Take the previous timestep
                    + availability[ts - 1][server_generation][dc]
                    # Subtract the expired servers based on life expectancy
                    - (
                        action_model[ts - sg_map[server_generation].life_expectancy][
                            dc
                        ][server_generation][Action.BUY]
                        if ts > sg_map[server_generation].life_expectancy
                        else 0
                    )
                )

    energy_cost = cp.new_int_var(0, INFINITY, "energy_cost")
    _ = cp.add(
        energy_cost
        == sum(
            (
                availability[ts][sg][dc]
                * sg_map[sg].energy_consumption
                * dc_map[dc].cost_of_energy
            )
            for ts in availability
            for sg in availability[ts]
            for dc in availability[ts][sg]
        )
    )

    maintenance_cost = cp.new_int_var(0, INFINITY, "maintenance_cost")
    _ = cp.add(
        maintenance_cost
        == sum(
            availability[ts][sg][dc] * sg_map[sg].average_maintenance_fee
            for ts in availability
            for sg in availability[ts]
            for dc in availability[ts][sg]
        )
    )

    for ts in availability:
        if ts == 0:
            continue
        for dc in datacenters:
            # Ensure we don't run out of slots on datacenters
            _ = cp.add(
                sum(
                    availability[ts][sg][dc.datacenter_id] * sg_map[sg].slots_size
                    for sg in availability[ts]
                )
                < dc_map[dc.datacenter_id].slots_capacity
            )

    # Calculate server utilization
    # This is the ratio of demand to availability for server type (sensitivity + server generation)
    revenues = {
        ts: {
            sg: {
                sen: cp.new_int_var(0, INFINITY, f"{ts}_{sg}_{sen}_util")
                for sen in Sensitivity
            }
            for sg in ServerGeneration
        }
        for ts in availability
    }
    revenues[0] = {
        sg: {sen: cp.new_int_var(0, 0, f"0_{sg}_{sen}_util") for sen in Sensitivity}
        for sg in ServerGeneration
    }

    for ts in revenues:
        if ts == 0:
            continue

        for sg in revenues[ts]:
            for sen in revenues[ts][sg]:
                total_availability = (
                    sum(
                        (
                            availability[ts][sg][dc.datacenter_id]
                            if dc.latency_sensitivity == sen
                            else 0
                        )
                        for dc in datacenters
                    )
                    * sg_map[sg].capacity
                )
                demand = cp.new_constant(demand_map[ts].get(sg, {sen: 0})[sen])
                # Get amount of demand that can be satisfied
                m = cp.new_int_var(0, INFINITY, f"{ts}_{sg}_{sen}_m")
                _ = cp.add_min_equality(
                    m,
                    [
                        demand,
                        total_availability * sg_map[sg].capacity,
                    ],  # Each server has *capacity* number of cpu/gpu that satisfies demand
                )
                _ = cp.add(revenues[ts][sg][sen] == m * sp_map[sg][sen])

    total_cost = cp.new_int_var(0, INFINITY, "total_cost")
    _ = cp.add(total_cost == buying_cost + energy_cost + maintenance_cost)
    total_revenue = sum(
        revenues[ts][sg][sen]
        for ts in revenues
        for sg in revenues[ts]
        for sen in revenues[ts][sg]
    )
    # Server utilization ratio of sum(min(demand, availability) / availability)/(len(servers) * len(Sensitivity))
    # To calculate this, we get the ratio of demand to availability at each timestamp
    # then we sum them up and divide by the number of timestamps
    utilization_ts = {
        ts: cp.new_int_var(0, 100, f"{ts}_util")
        for ts in range(1, max(demand.time_step for demand in demands) + 1)
    }
    utilization_fine = {
        ts: {
            dc.datacenter_id: {
                sg: cp.new_int_var(0, 100, f"{ts}_{dc.datacenter_id}_{sg}_util")
                for sg in ServerGeneration
            }
            for dc in datacenters
        }
        for ts in utilization_ts
    }
    for ts in utilization_fine:
        total_availability_ts = cp.new_int_var(0, INFINITY, f"{ts}_avail")
        _ = cp.add(
            total_availability_ts
            == sum(
                availability[ts][sg][dc.datacenter_id] * sg_map[sg].capacity
                for sg in ServerGeneration
                for dc in datacenters
            )
        )
        _ = cp.add_division_equality(
            utilization_ts[ts],
            total_availability_ts,
            len(datacenters) * len(ServerGeneration),
        )

        for dc in datacenters:
            for sg in ServerGeneration:
                # Get total demand for this timestamp
                demand_ts: int = (
                    demand_map.get(ts, {}).get(sg, {}).get(dc.latency_sensitivity, 0)
                )
                # Get total availability for this timestamp
                availability_ts = cp.new_int_var(
                    1, INFINITY, f"{ts}_{dc.datacenter_id}_{sg}_avail"
                )
                avail_sum = availability[ts][sg][dc.datacenter_id] * sg_map[sg].capacity

                if demand_ts == 0:
                    _ = cp.add(utilization_fine[ts][dc.datacenter_id][sg] == 0)
                else:
                    _ = cp.add(availability_ts == avail_sum)

                    m = cp.new_int_var(0, INFINITY, f"{ts}_min")
                    _ = cp.add_min_equality(m, [demand_ts, availability_ts])
                    _ = cp.add_division_equality(
                        utilization_fine[ts][dc.datacenter_id][sg],
                        m * 100,
                        availability_ts,
                    )
    # Server utilization ratio of sum(min(demand, availability) / availability)/(len(servers) * len(Sensitivity))
    # To calculate this, we get the ratio of demand to availability at each timestamp
    # then we sum them up and divide by the number of timestamps

    utilization_avg = cp.new_int_var(0, 100, "util_avg")
    total_utilization = cp.new_int_var(
        0,
        INFINITY,
        "total_util",
    )
    _ = cp.add(total_utilization == sum(utilization_ts[ts] for ts in utilization_ts))
    _ = cp.add_division_equality(
        utilization_avg,
        total_utilization,
        len(utilization_ts),
    )
    profit = cp.new_int_var(0, INFINITY, "profit")
    _ = cp.add(profit == total_revenue - total_cost)
    le_measure = cp.new_int_var(0, INFINITY, "le_measure")
    _ = cp.add_multiplication_equality(le_measure, [profit, utilization_avg])
    _ = cp.maximize(profit)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15 * 60
    status = solver.solve(cp)
    solution: list[SolutionEntry] = []
    if (
        status == cp_model.OPTIMAL  # type: ignore[reportUnnecessaryComparison]
        or status == cp_model.FEASIBLE  # type: ignore[reportUnnecessaryComparison]
    ):
        print(solver.solution_info())
        print(solver.response_stats())
        for ts in action_model:
            if ts == 0:
                continue
            for dc in action_model[ts]:
                for sg in action_model[ts][dc]:
                    for action in action_model[ts][dc][sg]:
                        val = solver.value(action_model[ts][dc][sg][action])
                        if val > 0:
                            print(f"{ts} {dc} {sg} {action} {val}")
                            solution.append(SolutionEntry(ts, dc, sg, action, val))
        # Ensure total availability at 0 is 0
        print("Profit:", solver.value(total_revenue) - solver.value(total_cost))
        print("Average Utilization:", solver.value(utilization_avg))
        print("Le Measure:", solver.value(le_measure))
        return solution
    else:
        print(solver.status_name(status))
        print(solver.solution_info())
        print(solver.response_stats())
        raise Exception("No solution found")
