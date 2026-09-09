from ortools.linear_solver import pywraplp
import time


def linear_solver(cfg, costs,
                  max_division=None,
                  new_detect=None,
                  extra_st= False,
                  ):
    """
    Solve the assignment problem using the SCIP solver.
    Args:
        costs:
        max_division: number of cells in division
        min_division: min number of cells in division
        extra_st: whether to add extra constraints
    Returns: list of assignments
    """
    # start_time = time.time()
    result = []
    num_framet0 = len(costs)  ## frame t + virtual node
    num_framet1 = len(costs[0])  ## frame t+1

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # # #  Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_framet0):
        for j in range(num_framet1):
            x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

    # # #  Constraints
    # 1 Each cell in frame t+1 is assigned to exactly one cell in frame t.
    for j in range(num_framet1):
        solver.Add(solver.Sum([x[i, j] for i in range(num_framet0)]) == 1)

    # 2 Each cell in frame t is assigned to at most 2 cells in frame t+1.
    if cfg.track.division:
        for i in range(num_framet0 - 1):
            solver.Add(solver.Sum([x[i, j] for j in range(num_framet1)]) <= 2)
    else:
        for i in range(num_framet0 - 1):
            solver.Add(solver.Sum([x[i, j] for j in range(num_framet1)]) <= 1)
        # solver.Add(solver.Sum([x[num_framet0 - 1, j] for j in range(num_framet1)]) >= max(0, num_framet1 - num_framet0))

    # extra constraints
    if extra_st:
        # # #  auxiliary variable
        division = [solver.IntVar(0, 1, f"d_{i}") for i in range(num_framet0 - 1)]
        # apoptosis = [solver.IntVar(0, 1, f"a_{i}") for i in range(num_framet0 - 1)]
        # movement = [solver.IntVar(0, 1, f"m_{i}") for i in range(num_framet0 - 1)]

        min_division: int = 20
        min_new_detect: int = 40
        division_ratio: float = 0.05
        new_detect_ratio: float = 0.05

        # # 3 division <= max_division_rate * num_framet0
        if max_division is None:
            max_division = max(num_framet0 * division_ratio + min_division, num_framet1 - num_framet0 + 1)  # virtual node=1
            # print('max_division:', division)
        for i in range(num_framet0 - 1):
            solver.Add(solver.Sum([x[i, j] for j in range(num_framet1)]) >= 2 * division[i])
            solver.Add(solver.Sum([x[i, j] for j in range(num_framet1)]) <= division[i] + 1)
        solver.Add(solver.Sum(division) <= max_division)

        # # 4 new_detect
        if new_detect is None:
            new_detect = max(new_detect_ratio*num_framet1+min_new_detect, num_framet1 - num_framet0)
            print(new_detect)
        solver.Add(solver.Sum([x[num_framet0 - 1, j] for j in range(num_framet1)]) <= new_detect)

        # # 5 Apoptosis or cells run in view <= max_apoptosis_rate * num_framet0
        # max_apoptosis = max(max_apoptosis, num_framet0 - num_framet1)
        # for i in range(num_framet0 - 1):
        #     solver.Add(solver.Sum([x[i, j] for j in range(num_framet1)]) <= 2 * (1 - apoptosis[i]))
        #     solver.Add(solver.Sum([x[i, j] for j in range(num_framet1)]) >= 1 - apoptosis[i])
        # solver.Add(solver.Sum(apoptosis) <= max_apoptosis)

        # # # 6 movement + apoptosis + division = num_framet0
        # for i in range(num_framet0 - 1):
        #     solver.Add(solver.Sum([x[i, j] for j in range(num_framet1)]) == movement[i] + 2 * division[i])
        #     solver.Add(movement[i] + apoptosis[i] + division[i] == 1)
        # solver.Add(solver.Sum(movement) >= (1-max_apoptosis_rate) * num_framet0)-10
        # # solver.Add(solver.Sum([movement[i] + division[i] for i in range(num_framet0-1)]) >=
        # #            np.ceil((1-max_apoptosis_rate) * num_framet0))

    # Objective
    objective_terms = []
    for i in range(num_framet0):
        for j in range(num_framet1):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Maximize(solver.Sum(objective_terms))

    # Solve
    # print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    # # print(f"Total cost = {-solver.Objective().Value()}\n")
        for j in range(num_framet1):
            for i in range(num_framet0):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.5:
                    # print(x[i, j].solution_value())
                    result.append(i)
                    # print(f"Worker {i} assigned to task {j}." + f" Cost: {costs[i][j]}")
    else:
        raise ValueError('No solution found.!')
    # print(f"Solve time: {time.time() - start_time}")
    return result


if __name__ == "__main__":
    costs = [
        [90, 80, 75, 70],
        [35, 85, 55, 65],
        [125, 95, 90, 95],
        [0, 0, 0, 0],
        # [45, 110, 95, 115],
        # [50, 100, 90, 100],
    ]
    linear_solver(costs)

    print('Model and solve a small MIP model with Python using Google\'s OR-Tools.')


