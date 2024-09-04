from collections import defaultdict

import numpy as np

import constants
import evaluation_v2
import plot
import reverse
from solver import models

seed = 2281
solution = reverse.get_solution("output/2281.json")

np.random.seed(seed)
evaluator = evaluation_v2.Evaluator(
    solution,
    constants.get_demand(),
    constants.get_servers(),
    constants.get_datacenters(),
    constants.get_selling_prices(),
)

plot.plot_capacity_demand(evaluator)


# Function to check for buys and dismisses within a range of timesteps
def check_buys_dismisses_in_range(
    actions: dict[int, list[models.SolutionEntry]], range_size: int = 5
):
    buys: dict[int, dict[models.ServerGeneration, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    dismisses: dict[int, dict[models.ServerGeneration, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for ts in actions:
        for entry in actions[ts]:
            if entry.action == models.Action.BUY:
                buys[ts][entry.server_generation] += entry.amount
            elif entry.action == models.Action.DISMISS:
                dismisses[ts][entry.server_generation] += entry.amount

    for ts in actions:
        for check_ts in range(ts, min(ts + range_size, max(actions.keys()) + 1)):
            for sg in models.ServerGeneration:
                if buys[ts][sg] != 0 and dismisses[check_ts][sg] != 0:
                    print(
                        f"Found buy at timestep {ts} and dismiss at timestep {check_ts} for generation {sg}"
                    )
                    print(
                        f"Buy amount: {buys[ts][sg]}, Dismiss amount: {dismisses[check_ts][sg]}"
                    )
                elif dismisses[ts][sg] != 0 and buys[check_ts][sg] != 0:
                    print(
                        f"Found dismiss at timestep {ts} and buy at timestep {check_ts} for generation {sg}"
                    )
                    print(
                        f"Dismiss amount: {dismisses[ts][sg]}, Buy amount: {buys[check_ts][sg]}"
                    )
        print(".", end="", flush=True)


# Run the analysis
# check_buys_dismisses_in_range(evaluator.actions, 20)
