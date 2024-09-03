# pyright: basic


import pandas as pd

from evaluation import get_actual_demand
from solver.models import Datacenter, Demand, SellingPrices, Server


def get_datacenters() -> list[Datacenter]:
    datacenters = pd.read_csv("data/datacenters.csv")

    return [
        Datacenter(**dc).setup()  # pyright: ignore[]
        for _, dc in datacenters.iterrows()
    ]


def get_servers() -> list[Server]:
    servers = pd.read_csv("data/servers.csv")

    return [
        Server(**server).setup()  # pyright: ignore[reportArgumentType]
        for _, server in servers.iterrows()
    ]


def get_selling_prices() -> list[SellingPrices]:
    selling_prices = pd.read_csv("data/selling_prices.csv")

    p = [
        SellingPrices(**sp).setup()  # pyright: ignore[]
        for _, sp in selling_prices.iterrows()
    ]
    return p


def get_demand() -> list[Demand]:
    parsed: list[Demand] = []

    for i, row in get_actual_demand(pd.read_csv("./data/demand.csv")).iterrows():
        parsed.append(
            Demand(
                row.time_step, row.server_generation, row.high, row.medium, row.low
            ).setup()  # pyright: ignore[reportUnknownArgumentType]
        )

    return parsed


if __name__ == "__main__":
    print(get_datacenters())
    print(get_servers())
    print(get_selling_prices())
