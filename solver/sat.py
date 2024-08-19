# pyright: basic
from ortools.sat.python import cp_model

from solver import models

from .models import Datacenter, Demand, SellingPrices, Server

t = "timestep"
d = "datacenter"
s = "server_generation"
a = "actions"
am = "amount"


def solve(
    demands: list[Demand],
    datacenters: list[Datacenter],
    selling_prices: list[SellingPrices],
    Servers: list[Server],
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
                        for action in ["buy", "sell"]
                    }
                    for server_generation in models.ServerGeneration
                }
            }
            for datacenter in datacenters
        }
        for timestep in range(0, max(demand.time_step for demand in demands) + 1)
    }
    cost = cp.new_int_var(0, 100_000_000, "cost")
    cp.add(
        cost
        == sum(
            model[t][d][s][a][am]
            for t in model
            for d in model[t]
            for s in model[t][d]
            for a in model[t][d][s]
        )
    )
    for ts in model:
        for dc in model[ts]:
            # Only one action (or less) can be taken per datacenter per timestep
            cp.add(
                (model[ts][dc][s]["buy"][am] > 0 + model[ts][dc][s]["sell"][am] > 0)
                != 2
            )
