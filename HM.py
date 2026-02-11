import sys
import os
import numpy as np
import gurobipy as gp
import networkx as nx
from itertools import product
from utils import direct_rev_prefs

# Solve Houtman and Mask Index.


# Class to hide Gurobi Prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def HM(p, x, method=3, cutoff=500):
#   - method is method to solve problem (method = 3 appears to work better)
#       method = 1 is by Linear Integer Programming to solve Minimum Feedback 
#                vertex Set Problem
#       method = 2 is using the method by Heufer and Hjertstrand
#       method = 3 is using the method by Demuynck and Rehbeck
#   - cutoff is maximum number of cycles to accept in method 1.
#       if there are too many cycles, method fails

    if method == 1:
        N = p.shape[0]
        DRP, DRSP = direct_rev_prefs(p, x)
        G = create_graph(DRSP)

        c1, c2 = get_all_cycles(G, cutoff)
        if c1:
            sol, index = HM_solution_LIP(list(c2), N)
        else:
            sol = []
            index = -1

    elif method == 2:
        e = np.dot(p, x.T)
        sol, index = HM_solution_HH(e)

    elif method == 3:
        e = np.dot(p, x.T)
        sol, index = HM_solution_DR(e)

    return index, sol


def create_graph(DRSP):
    N = DRSP.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i, j in cart_range(N):
        if DRSP[i, j]:
            G.add_edge(i, j)

    return G


def HM_solution_LIP(cycles, N):

    with HiddenPrints():
        # Declare model
        m = gp.Model()

        # Add variables (one per observation)
        x = m.addMVar(N, vtype=gp.GRB.BINARY, name="x")
        m.update()

        # Add constraints (one per cycle)
        nc = len(cycles)

        A = np.zeros([nc, N])

        for i in range(nc):
            for j in cycles[i]:
                A[i, j] = 1

        m.addMConstr(A, x, '>=', b=np.ones(nc))
        m.update()

        # Define objective function
        m.setMObjective(Q=None, c=np.ones(N), constant=0.0, xc=x,
                        sense=gp.GRB.MINIMIZE)
        m.update()

        # Solve model
        m.optimize()

    sol = np.argwhere(x.X > .5).flatten()

    index = len(sol) / N

    return sol, index


def HM_solution_HH(e):

    N = e.shape[0]

    # Define vector A in Theorem 1 of Heufer and Hjertstrand
    eps = 10 ** (-5)

    A = np.empty(N)
    for i in range(N):
        A[i] = e[i, i] + eps

    # Create the model and solve
    with HiddenPrints():
        # Declare model
        m = gp.Model()

        # Add variables v, u, psi
        v = m.addMVar(N, vtype=gp.GRB.BINARY, name='v')
        u = m.addMVar(N, lb=0.0, ub=(1 - eps), vtype=gp.GRB.CONTINUOUS,
                      name='u')
        psi = m.addMVar((N, N), vtype=gp.GRB.BINARY, name='psi')
        m.update()

        # Add constraints
        for i, j in cart_range(N):
            m.addConstr(u[i] - u[j] <= psi[i, j] - eps)
            m.addConstr(psi[i, j] - 1 <= u[i] - u[j])
            m.addConstr(v[i] * e[i, i] - e[i, j] <= psi[i, j] * A[i] - eps)
            m.addConstr((psi[i, j] - 1) * A[j] <= e[j, i] - v[j] * e[j, j])

        m.update()

        # Define objective function
        m.setMObjective(Q=None, c=np.ones(N), constant=0.0, xc=v,
                        sense=gp.GRB.MAXIMIZE)
        m.update()

        # Solve model
        m.optimize()

    sol = np.argwhere(v.X < .5).flatten()

    index = len(sol) / N

    return sol, index


def HM_solution_DR(e):
    # Houtman and Mask Index using solution in Demuynck and Rehbeck

    N = e.shape[0]

    # First, define epsilon and compute values of delta and alpha
    eps = 1 / (2 * N)
    alpha = 1.1 * max([e[i, i] for i in range(N)])
    delta = min([e[i, j] for i, j in cart_range(N)]) / 2
    for i, j in cart_range(N):
        if (e[i, j] > e[i, i]) and (e[i, j] - e[i, i] <= delta):
            delta = (e[i, j] - e[i, i]) / 2

    # Create model and solve
    with HiddenPrints():
        # Declare Model
        m = gp.Model()

        # Add variables u, A, U. To avoid confusion use V instead of U
        u = m.addMVar(N, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS,
                      name='u')
        A = m.addMVar(N, vtype=gp.GRB.BINARY, name='A')
        V = m.addMVar((N, N), vtype=gp.GRB.BINARY, name='V')

        m.update()

        # Add constraints
        for i, j in cart_range(N):
            m.addConstr(u[i] - u[j] + eps <= 2 * V[i, j])  # IP-1
            m.addConstr(V[i, j] - 1 <= u[i] - u[j])  # IP-2
            # IP-5 is written in the opposite direction to avoid error due to
            # left hand side being a numpy scalar
            # (see https://support.gurobi.com/hc/en-us/articles/360039628832-Constraint-has-no-bool-value-are-you-trying-lb-expr-ub-)
            m.addConstr(- delta + alpha * (V[i, j] + 1 - A[i]) >=
                        e[i, i] - e[i, j])  # IP-5
            m.addConstr(alpha * (V[j, i] + A[i] - 2) <= e[i, j] - e[i, i])  # IP-6

        m.update()

        # Add objective function
        m.setObjective(sum([A[i] for i in range(N)]), sense=gp.GRB.MAXIMIZE)

        m.update()

        # Solve Model
        m.optimize()

    # Get indices of removed observations
    sol = np.argwhere(A.X == 0).flatten()

    index = len(sol) / N

    return sol, index


def cart_range(N, N1=None):
    if N1:
        return product(range(N), range(N1))
    else:
        return product(range(N), range(N))


def get_all_cycles(g, x, cutoff=500):
    # G is graph of revealed preferences
    # x in data of choices

    from networkx import simple_cycles
    from itertools import islice

    ll = pairs_equal(x)

    success = True

    # Try to enumerate cutoff+1 simple cycles
    cycles = []
    for c in islice(simple_cycles(g), 0, cutoff + 1):
        garp_violation = True
        for pair in ll:
            if (pair[0] in c) and (pair[1] in c):
                garp_violation = False
                continue

        if garp_violation:
            cycles.append(c)

        if len(cycles) > cutoff:
            success = False
            break

    return success, cycles


def pairs_equal(x):
    N = x.shape[0]
    ll = []

    for i in range(N):
        ll.append((i, i))
        for j in range(i + 1, N):
            if np.array_equal(x[i], x[j]):
                ll.append((i, j))

    return ll
