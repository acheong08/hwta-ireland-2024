import numpy as np
        
class ServerTracker:

    def __init__(self) -> None:
        self.servers = [[0 for _ in range(21)] for _ in range(168)]

    def addServers(self, new_servers, day):
        for i in range(21):
            if new_servers[i] > 0:
                self.servers[day-1][i] += new_servers[i]
            
    
    def remove_old_servers(self, current_day):
        days_to_remove = current_day - 96
        if (days_to_remove < 0):
            pass
        else:
            for i in range(days_to_remove):
                self.servers[i] = [0 for _ in range(21)]
            
    def dismiss_servers(self, new_solution):
        # Ensure input sizes are correct
        if len(new_solution) != 21:
            raise ValueError("Input array A must be of length 21")
    
        # Iterate through each element in A
        for i, value in enumerate(new_solution):
            if value < 0:
                abs_value = abs(value)
                # Apply the subtraction to each element in the i-th sublist of server_array
                for j in range(168):
                    if abs_value <= 0:
                        break  # No more subtraction needed
                    if self.servers[i][j] > 0:
                        if self.servers[i][j] >= abs_value:
                            self.servers[i][j] -= abs_value
                            break  # We've fully applied the subtraction
                        else:
                            abs_value -= self.servers[i][j]
                            self.servers[i][j] = 0
    
    def update_all_servers(self, previous_results, new_results, day):
        self.addServers(new_results, day)
        self.dismiss_servers(new_results)
        self.remove_old_servers(current_day=day)        
    
    def map_servers_to_search_space(self):
        x = np.sum(self.servers, axis=0)
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
        
            




