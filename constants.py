# pyright: basic


import pandas as pd

from solver.models import Datacenter, SellingPrices, Server, Demand


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


if __name__ == "__main__":
    print(get_datacenters())
    print(get_servers())
    print(get_selling_prices())
