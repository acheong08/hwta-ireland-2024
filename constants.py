# pyright: basic


from re import IGNORECASE

import pandas as pd

from solver.models import Datacenter, SellingPrices, Server


def get_datacenters() -> list[Datacenter]:
    datacenters = pd.read_csv("data/datacenters.csv")

    return [Datacenter(**dc) for _, dc in datacenters.iterrows()]  # pyright: ignore[]


def get_servers() -> list[Server]:
    servers = pd.read_csv("data/servers.csv")

    return [
        Server(**server).setup()  # pyright: ignore[reportArgumentType]
        for _, server in servers.iterrows()
    ]


def get_selling_prices() -> list[SellingPrices]:
    selling_prices = pd.read_csv("data/selling_prices.csv")

    return [
        SellingPrices(**sp) for _, sp in selling_prices.iterrows()  # pyright: ignore[]
    ]


if __name__ == "__main__":
    print(get_datacenters())
    print(get_servers())
    print(get_selling_prices())
