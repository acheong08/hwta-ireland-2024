# pyright: basic
from ortools.sat.python import cp_model


def only_one():
    cp = cp_model.CpModel()

    x = cp.new_int_var(0, 10, "x")
    y = cp.new_int_var(0, 10, "y")

    xx = cp.new_int_var(0, 1, "xx")
    cp.add_modulo_equality(xx, x, 2)
    yy = cp.new_int_var(0, 1, "yy")
    cp.add_modulo_equality(yy, y, 2)
    cp.add(xx == 0).only_enforce_if(yy.is_equal_to(1))
    cp.add(yy == 0).only_enforce_if(xx.is_equal_to(1))

    solver = cp_model.CpSolver()
    status = solver.Solve(cp)
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        print(f"x={solver.Value(x)}, y={solver.Value(y)}")
    else:
        print("Status:", status)
        print("No solution found")

for i in range(32, 64):
    try:
        model = cp_model.CpModel()
        x = model.NewIntVar(0, 2**i, "x")
        y = model.NewIntVar(0, 10, "y")
        model.Add(x != y)
        model.Add(x == 2)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            print(f"x={solver.Value(x)}, y={solver.Value(y)}")
            print(f"x={solver.Value(x)}, y={solver.Value(y)}")
    except:
        print(f"Error: {i}")
        break
