from .solver.models import Action, ServerGeneration, SolutionEntry


def generate(entries: list[SolutionEntry]) -> list[dict[str, str | int]]:
    solution: list[dict[str, str | int]] = []
    # datacenter_id -> server_generation -> list[server id]
    ids: dict[str, dict[ServerGeneration, list[int]]] = {}
    counter = 0
    for entry in entries:
        if entry.action == Action.BUY:
            if ids.get(entry.datacenter_id) is None:
                ids[entry.datacenter_id] = {entry.server_generation: []}
            for _ in range(entry.amount):
                ids[entry.datacenter_id][entry.server_generation].append(counter)
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
                        "server_id": server_id,
                        "server_generation": entry.server_generation.value,
                        "action": "dismiss",
                    }
                )
    return solution
