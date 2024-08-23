# pyright: basic
from ortools.sat.python import cp_model

model = cp_model.CpModel()
solver = cp_model.CpSolver()

x = model.NewIntVar(0, 10, "x")
y = model.NewIntVar(0, 10, "y")
z = model.NewIntVar(0, 10**2, "z")
model.add_division_equality(z, sum([x, y]), z)
model.maximize(z)

status = solver.Solve(model)
if status == cp_model.OPTIMAL:
    print("x =", solver.Value(x))
    print("y =", solver.Value(y))
    print("z =", solver.Value(z))
else:
    print(solver.status_name(status))
    print(solver.response_stats())
    print(solver.solution_info())
