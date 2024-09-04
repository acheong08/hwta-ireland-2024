# pyright: reportUnknownMemberType=false, reportUnusedCallResult=false, reportUnknownArgumentType=false
import matplotlib.pyplot as plt

import constants
import solver.models as models
from evaluation_v2 import Evaluator


def plot_capacity_demand(evaluator: Evaluator):
    evaluator.plot_generation = models.ServerGeneration.CPU_S1
    _ = evaluator.get_score()

    plt.figure(figsize=(12, 6))
    colors = {"LOW": "blue", "MEDIUM": "green", "HIGH": "red"}
    datacenters = {dc.datacenter_id: dc for dc in constants.get_datacenters()}
    servers = {s.server_generation: s for s in constants.get_servers()}

    for sensitivity in models.Sensitivity:
        timesteps = range(1, 169)
        capacity = [evaluator.capacity_history[ts][sensitivity] for ts in timesteps]
        demand = [evaluator.demand_history[ts][sensitivity] for ts in timesteps]
        max_capacity = (
            sum(
                (
                    datacenters[dc].slots_capacity
                    / servers[evaluator.plot_generation].slots_size
                    if datacenters[dc].latency_sensitivity == sensitivity
                    else 0
                )
                for dc in datacenters
            )
            // 100
        )

        # Plot max capacity
        plt.plot(
            timesteps,
            [max_capacity] * len(timesteps),
            label=f"{sensitivity.name} Max Capacity",
            color=colors[sensitivity.name],
            linestyle=":",
        )

        # Plot capacity
        plt.plot(
            timesteps,
            capacity,
            label=f"{sensitivity.name} Capacity",
            color=colors[sensitivity.name],
            linestyle="-",
        )

        # Plot demand
        plt.plot(
            timesteps,
            demand,
            label=f"{sensitivity.name} Demand",
            color=colors[sensitivity.name],
            linestyle="--",
        )

    plt.xlabel("Timestep")
    plt.ylabel("Capacity / Demand")
    plt.title(
        f"Server Capacity and Demand Over Time for {evaluator.plot_generation.name}"
    )
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
