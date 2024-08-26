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
                server_generation: cp.new_int_var(
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
                    f"{timestep}_{datacenter}_{server_generation}_action",
                )
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
            action_model[t][d][s] * sg_map[s].purchase_price
            for t in action_model
            for d in action_model[t]
            for s in action_model[t][d]
        )
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
                    == (action_model[ts][dc][server_generation])
                    # Take the previous timestep
                    + availability[ts - 1][server_generation][dc]
                    # Subtract the expired servers based on life expectancy
                    - (
                        action_model[
                            ts - sg_map[server_generation].life_expectancy + 1
                        ][dc][server_generation]
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
                sen: cp.new_int_var(0, INFINITY, f"{ts}_{sg}_{sen}_rev")
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
                total_availability = sum(
                    (
                        availability[ts][sg][dc.datacenter_id]
                        if dc.latency_sensitivity == sen
                        else 0
                    )
                    for dc in datacenters
                ) * int(sg_map[sg].capacity / sg_map[sg].slots_size)
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
    cp.maximize(total_revenue - total_cost)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5 * 60
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
                    val = solver.value(action_model[ts][dc][sg])
                    if val > 0:
                        # print(f"{ts} {dc} {sg} {action} {val}")
                        solution.append(SolutionEntry(ts, dc, sg, Action.BUY, val))
        # total_profit = 0
        # for ts in action_model:
        #     # Calculate revenue
        #     revenue = sum(
        #         solver.value(revenues[ts][sg][sen])
        #         for sg in revenues[ts]
        #         for sen in revenues[ts][sg]
        #     )
        #     # Calculate maintenance cost
        #     maintenance = sum(
        #         solver.value(availability[ts][sg][dc])
        #         * sg_map[sg].average_maintenance_fee
        #         for sg in availability[ts]
        #         for dc in availability[ts][sg]
        #     )
        #     # Calculate energy cost
        #     energy = sum(
        #         solver.value(availability[ts][sg][dc])
        #         * sg_map[sg].energy_consumption
        #         * dc_map[dc].cost_of_energy
        #         for sg in availability[ts]
        #         for dc in availability[ts][sg]
        #     )
        #     # Buying costs
        #     buying = sum(
        #         solver.value(action_model[ts][dc][sg]) * sg_map[sg].purchase_price
        #         for dc in action_model[ts]
        #         for sg in action_model[ts][dc]
        #     )
        #
        #     print(f"{ts} -R:{revenue/100} C:{(maintenance+energy+buying)/100}")
        #     total_profit += (revenue / 100) - ((maintenance + energy + buying) / 100)
        # print(total_profit, solver.value(total_revenue), solver.value(total_cost))
        print(solver.value(total_revenue) / 100 - solver.value(total_cost) / 100)

        return solution
    else:
        print(solver.status_name(status))
        print(solver.solution_info())
        print(solver.response_stats())
        raise Exception("No solution found")
