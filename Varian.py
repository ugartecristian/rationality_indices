import sys
import os
import numpy as np
import gurobipy as gp
from time import time
from utils import cart_range, garp


# Class to hide Gurobi Prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def Varian_DR(p, x, objective='linear'):
    # Varian Index using solution in Demuynck and Rehbeck
    # objective is the loss function to minimize, it can be
    #       linear: sum_{k} (1-v_k)
    #       quadratic: sqrt(sum_{k} (1-v_k)^2)

    time_start = time()

    N = p.shape[0]

    # First check if data satisfies GARP to avoid entering maximization problem
    if garp(p, x):
        duration = time() - time_start
        return 0, np.ones(N), duration

    else:
        e = np.dot(p, x.T)

        # Define epsilon and compute values of delta and alpha
        eps = 1 / (2 * N)
        alpha = 1.1 * max([e[i, i] for i in range(N)])

        # Create model and solve
        with HiddenPrints():
            pass

            # Declare Model
            m = gp.Model()

            # Add variables u, v, and U.
            #   - U and u follow the notation in the paper
            #   - v is the vector of the Varian Index
            u = m.addMVar(N, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name='u')
            v = m.addMVar(N, lb=0.0, ub=1.0, obj=1.0, vtype=gp.GRB.CONTINUOUS,
                          name='v')
            U = m.addMVar((N, N), vtype=gp.GRB.BINARY, name='U')

            m.update()

            # Add constraints
            for i, j in cart_range(N):
                # IP-1
                m.addConstr(u[i] - u[j] <= - eps + 2 * U[i, j])
                # IP-2
                m.addConstr(U[i, j] - 1 <= u[i] - u[j])
                # IP-7
                m.addConstr(v[i] * e[i, i] - e[i, j] <= alpha * U[i, j])
                # IP-8
                m.addConstr(alpha * (U[j, i] - 1) <= e[i, j] - v[i] * e[i, i])

            m.update()

            # Add objective function
            if objective == 'linear':
                m.setObjective(sum([(1 - v[i]) / N for i in range(N)]),
                               sense=gp.GRB.MINIMIZE)
            elif objective == 'quadratic':
                m.setObjective(sum([(1 - v[i]) * (1 - v[i]) / N
                                    for i in range(N)]),
                               sense=gp.GRB.MINIMIZE)
            else:
                raise ValueError("Objective function must be linear or \
                                  quadratic")

            m.update()

            # Solve Model
            m.optimize()

        # Get solution
        sol = v.X

        index = m.getObjective().getValue()

        total_time = time() - time_start

        return index, sol, total_time
