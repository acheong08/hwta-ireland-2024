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
    entry_map = dict[
        int, dict[ServerGeneration, dict[str, dict[Action, tuple[int, str]]]]
    ]()
    for entry in entries:
        if entry_map.get(entry.timestep) is None:
            entry_map[entry.timestep] = {}
        if entry_map[entry.timestep].get(entry.server_generation) is None:
            entry_map[entry.timestep][entry.server_generation] = {}
        if (
            entry_map[entry.timestep][entry.server_generation].get(entry.datacenter_id)
            is None
        ):
            entry_map[entry.timestep][entry.server_generation][entry.datacenter_id] = {
                Action.BUY: (0, ""),
                Action.DISMISS: (0, ""),
                Action.MOVE: (0, ""),
            }
        entry_map[entry.timestep][entry.server_generation][entry.datacenter_id][
            entry.action
        ] = (entry.amount, entry.datacenter_target)
    counter = 0
    for ts in range(1, 169):
        if entry_map.get(ts) is None:
            continue
        for server_generation, datacenter_map in entry_map[ts].items():
            for datacenter_id, action_map in datacenter_map.items():
                amount = action_map[Action.BUY][0]
                if ids.get(datacenter_id) is None:
                    ids[datacenter_id] = {server_generation: []}
                if ids[datacenter_id].get(server_generation) is None:
                    ids[datacenter_id][server_generation] = []
                for _ in range(amount):
                    ids[datacenter_id][server_generation].append(
                        {
                            "id": counter,
                            "expires_at": ts
                            + server_map[server_generation].life_expectancy
                            - 1,
                        }
                    )
                    solution.append(
                        {
                            "time_step": ts,
                            "datacenter_id": datacenter_id,
                            "server_id": counter,
                            "server_generation": server_generation.value,
                            "action": "buy",
                        }
                    )
                    counter += 1
                # # Pop until we have no more expired servers
                while (
                    len(ids[datacenter_id][server_generation]) > 0
                    and ids[datacenter_id][server_generation][0]["expires_at"] <= ts
                ):
                    _ = ids[datacenter_id][server_generation].pop(0)
                amount = action_map[Action.DISMISS][0]
                for _ in range(amount):
                    if not ids[datacenter_id][server_generation]:
                        break
                    server_id = ids[datacenter_id][server_generation].pop(0)
                    solution.append(
                        {
                            "time_step": ts,
                            "datacenter_id": datacenter_id,
                            "server_id": server_id["id"],
                            "server_generation": server_generation.value,
                            "action": "dismiss",
                        }
                    )
                amount = action_map[Action.MOVE][0]
                target_datacenter = action_map[Action.MOVE][1]
                if ids.get(target_datacenter) is None:
                    ids[target_datacenter] = {server_generation: []}
                if ids[target_datacenter].get(server_generation) is None:
                    ids[target_datacenter][server_generation] = []
                for _ in range(amount):
                    if not ids[datacenter_id][server_generation]:
                        break
                    server_id = ids[datacenter_id][server_generation].pop(0)
                    solution.append(
                        {
                            "time_step": ts,
                            "datacenter_id": datacenter_id,
                            "server_id": server_id["id"],
                            "server_generation": server_generation.value,
                            "action": "move",
                        }
                    )
                    solution.append(
                        {
                            "time_step": ts + 1,
                            "datacenter_id": target_datacenter,
                            "server_id": server_id["id"],
                            "server_generation": server_generation.value,
                            "action": "hold",
                        }
                    )
                    if ids.get(target_datacenter) is None:
                        ids[target_datacenter] = {server_generation: []}
                    if ids[target_datacenter].get(server_generation) is None:
                        ids[target_datacenter][server_generation] = []
                    ids[target_datacenter][server_generation].insert(0, server_id)
                    ids[target_datacenter][server_generation] = sorted(
                        ids[target_datacenter][server_generation],
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
