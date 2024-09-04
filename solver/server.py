import numpy as np

class ServerTracker:

    def __init__(self, types) -> None:
        self.types = types
        self.servers =[[] for _ in range(types)]
        self.x = [0 for _ in range(21)]

    def add_server(self, type, day):
        self.servers[type].append(day)
    
    def update_all_servers(self, previous_day, current_day, day):
        new_servers = np.array(current_day) - np.array(previous_day)
        for i in range(len(new_servers)):
            val = 2 if (i % 7 < 4) else 4
            if new_servers[i] < 0:
                for j in range(new_servers[i] // val):
                    self.removeServer(i)
            elif new_servers[i] > 0:
                for j in range(new_servers[i] // val):
                    self.add_server(i, day)
        
    
    def remove_old_servers(self, current_day):
        for i in range(len(self.servers)):
            for j in range(len(self.servers[i])):
                if (current_day - self.servers[i][j] >= 96):
                    self.servers.remove(self.servers[i][j])

    
    def removeServer(self, type):
        self.server[type].remove(min(self.server[type]))
        
    
    def map_servers_to_search_space(self):
        x = [0 for _ in range(self.types)]
        for i in range(len(self.servers)):  
            if i % 7 > 3:
                total_sum = np.sum(self.servers[i]) * 4
            else:
                total_sum = np.sum(self.servers[i]) * 2
            x[i] = total_sum
        
        return x
    
    def define_restrictions(self, x):
        restrictions = [[], []]
        group_sums = [25245, 15300, 15300]
        # restrictions would look like this: [-ranges, +ranges]

        for i in range(len(x)):
            rangeVal = group_sums[i // 7] - x[i]
            restrictions[0].append(-x[i])
            restrictions[1].append(rangeVal)

        # restrictions define xl and xu for the next iteration

        return restrictions
        

        
            




