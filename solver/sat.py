# pyright: basic
from ortools.sat.python import cp_model

from .models import Datacenter, Demand, SellingPrices, Server


def solve(
    demands: list[Demand],
    datacenters: list[Datacenter],
    selling_prices: list[SellingPrices],
    Servers: list[Server],
) -> None:
    model = cp_model.CpModel()
    # Just an example. We should figure out the algebra on paper first
    for timestep in range(0, max([demand.time_step for demand in demands])):
        for datacenter in datacenters:
            buy = model.new_int_var(0, 1, f"{datacenter.datacenter_id}_{timestep}_buy")
            sell = model.new_int_var(
                0, 1, f"{datacenter.datacenter_id}_{timestep}_sell"
            )
            idle = model.new_int_var(
                0, 1, f"{datacenter.datacenter_id}_{timestep}_idle"
            )
            _ = model.add(sum([buy, sell, idle]) == 1)
