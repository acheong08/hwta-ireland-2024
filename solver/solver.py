import numpy as np

from solver.models import (
    Action,
    Datacenter,
    Demand,
    SellingPrices,
    Sensitivity,
    Server,
    ServerGeneration,
    SolutionEntry,
)

from .grad_descent import improved_gradient_descent
from .helper import mapDemandToVector, mapSellingPriceToVector
    
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

    d_vector = mapDemandToVector(demand_map[1])
    z_vector = [100 for _ in range(21)]
    p_vector = mapSellingPriceToVector(sp_map)

    z_vector = improved_gradient_descent(z_vector, d_vector, p_vector)
    




