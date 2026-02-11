import networkx as nx
import numpy as np
import sys
import os
from time import time
from functions.grb_lazy import solve_problem as MFAS
from utils import direct_rev_prefs, cart_range, to_unique, garp, tran_clos, \
                  dom_weak


# Class to hide Prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def MM(p, x):
    # Receives dataset p, x and returns
    #   - MM Index
    #   - list of "misakes"
    #   - time it took to compute
    start = time()

    # Obtain revealed preferences, and convert to data of unique choices
    DRP, DRSP = direct_rev_prefs(p, x)
    x1, DRP1, DRSP1 = to_unique(x, DRP, DRSP)
    G = create_graph(DRSP1, x1)

    # If DRSP1 is acyclic, MM Index = 0
    if nx.is_directed_acyclic_graph(G):
        tt = time() - start
        return 0, [], tt

    else:
        with HiddenPrints():
            M, cost, A = MFAS(G)
        mm = cost / sum(sum(DRSP1))

        # generate list of mistakes in original observations
        M_final = []
        for m in M:
            for i, j in cart_range(DRP.shape[0]):
                if np.array_equal(x1[m[0]], x[i]) and \
                   np.array_equal(x1[m[1]], x[j]):
                    M_final.append((i, j))

        tt = time() - start
        return mm, M_final, tt


def create_graph(RP, x):
    # Create graph from revealed preferences
    # x is used to assign weights

    N = RP.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i, j in cart_range(N):
        if RP[i, j]:
            if dom_weak(x[i], x[j]):
                G.add_edge(i, j, weight=N ** 2, orig_edges=[(i, j)])
            else:
                G.add_edge(i, j, weight=1, orig_edges=[(i, j)])

    return G


def mm_tuple(p, x):
    # Receive dataset p, x and returns MMI and tuple of Mistakes (M^w, M^s)
    # Tuple is returned as list of two lists

    N = p.shape[0]

    # If GARP holds, then no mistakes
    if garp(p, x):
        return 0, [[], []]

    else:
        # first get direct revealed prefs and Ms
        DRP, DRSP = direct_rev_prefs(p, x)
        for i, j in cart_range(N):
            if DRSP[i, j] and not DRP[i, j]:
                raise ValueError(f'Error in revealed preferences: {i} \
                                   strictly revealed preferred to {j} but not \
                                   revealed preferred')
        mmi, Ms, t = mm_index(p, x)

        # Get set C
        C = []
        for i, j in cart_range(N):
            if dom_weak(x[i], x[j]) or (DRSP[i, j] and (i, j) not in Ms):
                C.append((i, j))

        Mw = []
        for i, j in cart_range(N):
            if DRP[i, j] and ((i, j) not in C):
                Mw.append((i, j))

        Mw0 = Mw.copy()

        for e in Mw0:
            Mw1 = Mw.copy()
            Mw1.remove(e)

            if garpM(DRP, DRSP, Mw1, Ms):
                Mw.remove(e)
                # print(f'    Edge {e} removed')

            else:
                pass

        return mmi, [Mw, Ms]

        # INCOMPLETE - So far we have only defined Mw_0 in Algorithm 1 of paper


def garpM(DRP, DRSP, Mw, Ms):
    # Receives boolean matrices of directly revealed preferences DRP and
    #   directly revealed strict preferences DRSP and lists of mistakes Mw from
    #   weak preferences and Ms from strict preferences
    #   ((Mw, Ms) form a tuple of mistakes)
    #
    # Returns True if data satisfies GARP_M, and zero otherwise

    DRP1 = np.copy(DRP)
    DRSP1 = np.copy(DRSP)

    for m in Mw:
        DRP1[m[0], m[1]] = False
    for m in Ms:
        DRSP1[m[0], m[1]] = False

    RP = tran_clos(DRP1)

    garp = True
    for i, j in cart_range(RP.shape[0]):
        if RP[i, j] and DRSP1[j, i]:
            garp = False
            break

    return garp
