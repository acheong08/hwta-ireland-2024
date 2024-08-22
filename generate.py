from constants import get_servers
from solver.models import Action, Server, ServerGeneration, SolutionEntry
from utils import save_solution  # type: ignore[import]


def generate(
    entries: list[SolutionEntry], servers: list[Server]
) -> list[dict[str, str | int]]:
    server_map = {server.server_generation: server for server in servers}
    solution: list[dict[str, str | int]] = []
    # datacenter_id -> server_generation -> list[server id]
    ids: dict[str, dict[ServerGeneration, list[dict[str, int]]]] = {}
    counter = 0
    for entry in entries:
        if entry.action == Action.BUY:
            if ids.get(entry.datacenter_id) is None:
                ids[entry.datacenter_id] = {entry.server_generation: []}
            if ids[entry.datacenter_id].get(entry.server_generation) is None:
                ids[entry.datacenter_id][entry.server_generation] = []
            for _ in range(entry.amount * 1000):
                ids[entry.datacenter_id][entry.server_generation].append(
                    {
                        "id": counter,
                        "expires_at": entry.timestep
                        + server_map[entry.server_generation].life_expectancy,
                    }
                )
                solution.append(
                    {
                        "time_step": entry.timestep,
                        "datacenter_id": entry.datacenter_id,
                        "server_id": counter,
                        "server_generation": entry.server_generation.value,
                        "action": "buy",
                    }
                )
                counter += 1
        elif entry.action == Action.DISMISS:
            # Pop until we have no more expired servers
            while (
                ids[entry.datacenter_id][entry.server_generation][0]["expires_at"]
                <= entry.timestep
            ):
                _ = ids[entry.datacenter_id][entry.server_generation].pop(0)
            for _ in range(entry.amount * 1000):
                server_id = ids[entry.datacenter_id][entry.server_generation].pop(0)
                solution.append(
                    {
                        "time_step": entry.timestep,
                        "datacenter_id": entry.datacenter_id,
                        "server_id": server_id["id"],
                        "server_generation": entry.server_generation.value,
                        "action": "dismiss",
                    }
                )
    return solution


if __name__ == "__main__":
    entries = [
        SolutionEntry(1, "dc1", ServerGeneration.GPU_S1, Action.BUY, 10),
        SolutionEntry(1, "dc2", ServerGeneration.GPU_S2, Action.BUY, 3),
        SolutionEntry(2, "dc1", ServerGeneration.GPU_S1, Action.DISMISS, 5),
        SolutionEntry(2, "dc2", ServerGeneration.GPU_S2, Action.DISMISS, 3),
        SolutionEntry(3, "dc1", ServerGeneration.GPU_S1, Action.DISMISS, 5),
    ]

    save_solution(generate(entries, get_servers()), "data/test.json")