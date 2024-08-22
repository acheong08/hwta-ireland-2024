# pyright: basic
from constants import get_datacenters, get_servers

for ts in range(1, 169):
    dc_map = {dc.datacenter_id: dc for dc in get_datacenters()}
    sg_map = {sg.server_generation: sg for sg in get_servers()}
    print(ts, ":", end=" ")
    for dc in get_datacenters():
        for server in get_servers():
            capacity = (
                dc_map[dc.datacenter_id].slots_capacity
                // sg_map[server.server_generation].slots_size
            )
            release_time = sg_map[server.server_generation].release_time
            # If within release time
            if release_time[1] >= ts >= release_time[0]:
                print(capacity, end=" ")
            else:
                print(0, end=" ")
    print()
