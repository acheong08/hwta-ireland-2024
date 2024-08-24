from constants import get_servers
from solver.models import Action, Server, ServerGeneration, SolutionEntry
from utils import save_solution  # type: ignore[import]


class Solution:
    redone: list[SolutionEntry] = []
    internal: list[dict[str, str | int]]
    counter: int = 0
    # Datacenter, server generation, list of server ids with expiration
    ids: dict[str, dict[ServerGeneration, list[dict[str, int]]]]
    servers_map: dict[ServerGeneration, Server]

    def __init__(self, servers: list[Server]):
        self.internal = []
        self.ids = {}
        self.servers_map = {server.server_generation: server for server in servers}

    def ensure_id_exists(self, datacenter_id: str, server_generation: ServerGeneration):
        if self.ids.get(datacenter_id) is None:
            self.ids[datacenter_id] = {server_generation: []}
        if self.ids[datacenter_id].get(server_generation) is None:
            self.ids[datacenter_id][server_generation] = []

    def sufficient_ids(
        self, datacenter_id: str, server_generation: ServerGeneration, amount: int
    ):
        return len(self.ids[datacenter_id][server_generation]) >= amount

    def add(self, entry: SolutionEntry):
        self.redone.append(entry)
        if entry.action == Action.BUY:
            for _ in range(entry.amount):
                self.ids[entry.datacenter_id][entry.server_generation].append(
                    {
                        "id": self.counter,
                        "expires_at": entry.timestep
                        + self.servers_map[entry.server_generation].life_expectancy,
                    }
                )
                self.internal.append(
                    {
                        "time_step": entry.timestep,
                        "datacenter_id": entry.datacenter_id,
                        "server_id": self.counter,
                        "server_generation": entry.server_generation.value,
                        "action": "buy",
                    }
                )
                self.counter += 1
        elif entry.action == Action.DISMISS:
            for _ in range(entry.amount):
                server_id = self.ids[entry.datacenter_id][entry.server_generation].pop(
                    0
                )
                self.internal.append(
                    {
                        "time_step": entry.timestep,
                        "datacenter_id": entry.datacenter_id,
                        "server_id": server_id["id"],
                        "server_generation": entry.server_generation.value,
                        "action": "dismiss",
                    }
                )

    def generate(self, entries: list[SolutionEntry]):
        # Timestep -> datacenter_id -> action -> [SolutionEntry]
        entry_map: dict[
            int, dict[str, dict[ServerGeneration, dict[Action, SolutionEntry]]]
        ] = {}
        for entry in entries:
            if entry_map.get(entry.timestep) is None:
                entry_map[entry.timestep] = {}
            if entry_map[entry.timestep].get(entry.datacenter_id) is None:
                entry_map[entry.timestep][entry.datacenter_id] = {}
            if (
                entry_map[entry.timestep][entry.datacenter_id].get(
                    entry.server_generation
                )
                is None
            ):
                entry_map[entry.timestep][entry.datacenter_id][
                    entry.server_generation
                ] = {}
            if (
                entry_map[entry.timestep][entry.datacenter_id][
                    entry.server_generation
                ].get(entry.action)
                is not None
            ):
                raise Exception("Duplicate entry")
            entry_map[entry.timestep][entry.datacenter_id][entry.server_generation][
                entry.action
            ] = entry

        # For each timestep/datacenter, we dismiss until there is nothing left, then cancel out the remaining buys/dismiss, then buy
        for ts in entry_map:
            # Remove expired servers from ids
            for datacenter_id in self.ids:
                for server_generation in self.ids[datacenter_id]:
                    for server in self.ids[datacenter_id][server_generation]:
                        if server["expires_at"] < ts:
                            self.ids[datacenter_id][server_generation].remove(server)
            for datacenter_id in entry_map[ts]:
                dismissals: list[SolutionEntry] = []
                buys: dict[ServerGeneration, SolutionEntry] = {}
                for server_generation in entry_map[ts][datacenter_id]:
                    self.ensure_id_exists(datacenter_id, server_generation)
                    if entry_map[ts][datacenter_id][server_generation].get(
                        Action.DISMISS
                    ):
                        dismissals.append(
                            entry_map[ts][datacenter_id][server_generation][
                                Action.DISMISS
                            ]
                        )
                    if entry_map[ts][datacenter_id][server_generation].get(Action.BUY):
                        buys[server_generation] = entry_map[ts][datacenter_id][
                            server_generation
                        ][Action.BUY]
                for dismissal in dismissals:
                    if self.sufficient_ids(
                        dismissal.datacenter_id,
                        dismissal.server_generation,
                        dismissal.amount,
                    ):
                        self.add(dismissal)
                        continue
                    if buys.get(dismissal.server_generation) is None:
                        raise Exception("No buys to cancel out excess dismissals")
                    if buys[dismissal.server_generation].amount < dismissal.amount:
                        raise Exception("Not enough buys to cancel out dismissals")
                    # Dismiss as much as possible
                    leftover = len(self.ids[datacenter_id][dismissal.server_generation])
                    diff = dismissal.amount - leftover
                    dismissal.amount = leftover
                    self.add(dismissal)
                    buys[dismissal.server_generation].amount -= diff

                for server_generation in buys:
                    self.add(buys[server_generation])

    def solution(self):
        # if len(self.redone) != len(self.internal):
        #     raise Exception("Mismatch between redone and internal")
        return self.internal


def generate(entries: list[SolutionEntry], servers: list[Server]) -> Solution:
    entry_map: dict[int, list[SolutionEntry]] = {}
    for entry in entries:
        if entry_map.get(entry.timestep) is None:
            entry_map[entry.timestep] = []
        entry_map[entry.timestep].append(entry)
    solution = Solution(servers)
    solution.generate(entries)
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
    solution = generate(entries, get_servers())
    for entry in solution.redone:
        print(
            f"{entry.timestep} {entry.datacenter_id} {entry.server_generation} {entry.action} {entry.amount}"
        )
    save_solution(solution.solution(), "output/test.json")
