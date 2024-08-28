# pyright: basic
from scipy.stats import truncweibull_min

MAX = 1000000

total = 0

for i in range(0, MAX):
    total += truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()
print(total / MAX)
