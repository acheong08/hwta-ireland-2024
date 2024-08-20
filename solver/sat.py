# pyright: reportAssignmentType=false
from ortools.sat.python import cp_model

from solver import models

from .models import Datacenter, Demand, SellingPrices, Server

t = "timestep"
d = "datacenter"
s = "server_generation"
a = "actions"
am = "amount"
actions = ["buy", "sell"]


def solve(
    demands: list[Demand],
    datacenters: list[Datacenter],
    selling_prices: list[SellingPrices],
    servers: list[Server],
) -> None:

    cp = cp_model.CpModel()
    # Just an example. We should figure out the algebra on paper first
    model = {
        t: {
            d: {
                s: {
                    a: {
                        am: cp.new_int_var(
                            0,
                            100_000_000,
                            f"{timestep}_{datacenter}_{server_generation}_{action}",
                        )
                        for action in actions
                    }
                    for server_generation in models.ServerGeneration
                }
            }
            for datacenter in datacenters
        }
        for timestep in range(0, max(demand.time_step for demand in demands) + 1)
    }
    server_costs: dict[str, Server] = {
        server.server_generation: server for server in servers
    }
    # We create a map of server costs so we don't need to search
    buying_cost = cp.new_int_var(0, 100_000_000, "cost")
    _ = cp.add(
        buying_cost
        == sum(
            model[t][d][s][a][am] * server_costs[s].purchase_price
            for t in model
            for d in model[t]
            for s in model[t][d]
            for a in model[t][d][s]
        )
    )
    for ts in model:
        for dc in model[ts]:
            # Only one action (or less) can be taken per datacenter per timestep
            for action in actions:
                cur_mod = cp.new_int_var(0, 1, f"{ts}_{dc}_{action}")
                _ = cp.add_modulo_equality(cur_mod, model[ts][dc][s][action][am], 2)
                for other_action in actions:
                    if other_action != action:
                        ot_mod = cp.new_int_var(
                            0, 1, f"{ts}_{dc}_{action}_{other_action}"
                        )
                        _ = cp.add_modulo_equality(
                            ot_mod, model[ts][dc][s][other_action][am], 2
                        )
                        _ = cp.add(cur_mod == 0).only_enforce_if(ot_mod.is_equal_to(1))
