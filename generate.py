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
            for _ in range(entry.amount):
                ids[entry.datacenter_id][entry.server_generation].append(
                    {
                        "id": counter,
                        "expires_at": entry.timestep
                        + server_map[entry.server_generation].life_expectancy
                        - 1,
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

            for _ in range(entry.amount):
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
            # # Pop until we have no more expired servers
            while (
                len(ids[entry.datacenter_id][entry.server_generation]) > 0
                and ids[entry.datacenter_id][entry.server_generation][0]["expires_at"]
                <= entry.timestep
            ):
                _ = ids[entry.datacenter_id][entry.server_generation].pop(0)
        elif entry.action == Action.MOVE:
            if ids.get(entry.datacenter_target) is None:
                ids[entry.datacenter_target] = {entry.server_generation: []}
            if ids[entry.datacenter_target].get(entry.server_generation) is None:
                ids[entry.datacenter_target][entry.server_generation] = []

            for _ in range(entry.amount):
                # Find oldest server id. This should be at the front of the list
                server_id = ids[entry.datacenter_id][entry.server_generation].pop(0)
                ids[entry.datacenter_target][entry.server_generation].append(server_id)
                solution.append(
                    {
                        "time_step": entry.timestep,
                        "datacenter_id": entry.datacenter_id,
                        "server_id": server_id["id"],
                        "server_generation": entry.server_generation.value,
                        "action": "move",
                    }
                )
                # Now hold it at the new datacenter
                solution.append(
                    {
                        "time_step": entry.timestep + 1,
                        "datacenter_id": entry.datacenter_target,
                        "server_id": server_id["id"],
                        "server_generation": entry.server_generation.value,
                        "action": "hold",
                    }
                )
                # Now we put the id back to the new datacenter
                ids[entry.datacenter_target][entry.server_generation].insert(
                    0, server_id
                )
                # Sort the list by server id
                ids[entry.datacenter_target][entry.server_generation] = sorted(
                    ids[entry.datacenter_target][entry.server_generation],
                    key=lambda x: x["id"],
                )
    return solution


if __name__ == "__main__":
    solution = open("solution.txt", "r")
    entries: list[SolutionEntry] = []
    for line in solution.readlines():
        parts = line.split()
        action = "buy" if parts[3] == "Action.BUY" else "dismiss"
        server_generation = parts[2].split(".")[-1].replace("_", ".")
        entries.append(
            SolutionEntry(
                timestep=int(parts[0]),
                datacenter_id=parts[1],
                server_generation=ServerGeneration(server_generation),
                action=Action(action),
                amount=int(parts[4]),
            )
        )
    save_solution(generate(entries, get_servers()), "output/test.json")
