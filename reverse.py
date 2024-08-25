import json

import seeds
from constants import get_datacenters, get_servers
from generate import generate
from solver.models import Action, ServerGeneration, SolutionEntry
from utils import save_solution  # type: ignore[import]

data: list[dict[str, int | str]] = json.load(open("data/solution_example.json"))

aggregate: dict[int, dict[str, dict[ServerGeneration, int]]] = {
    ts: {
        dc.datacenter_id: {gen: 0 for gen in ServerGeneration}
        for dc in get_datacenters()
    }
    for ts in range(1, 168)
}


for entry in data:
    aggregate[int(entry["time_step"])][str(entry["datacenter_id"])][
        ServerGeneration(entry["server_generation"])
    ] += 1

solutions: list[SolutionEntry] = []

for ts in aggregate:
    for dc in aggregate[ts]:
        for gen in aggregate[ts][dc]:
            count = aggregate[ts][dc][gen]
            if count == 0:
                continue
            solutions.append(SolutionEntry(ts, dc, gen, Action.BUY, count))


def get_solution():
    return generate(solutions, get_servers())


if __name__ == "__main__":
    for seed in seeds.known_seeds("training"):
        solutions[-1].amount += 1
        save_solution(generate(solutions, get_servers()), f"./output/{seed}.json")
