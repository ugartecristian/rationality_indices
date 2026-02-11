import numpy as np
import pandas as pd
from itertools import product, permutations, islice
from networkx import simple_cycles
from tarjan.tc import tc


def cart_range(N, N1=None):
    if N1:
        return product(range(N), range(N1))
    else:
        return product(range(N), range(N))


def direct_rev_prefs(p, x, p_e=1, crit='MON'):
    e = np.dot(p, x.T)
    N = p.shape[0]

    expen = [[e[i, i] * p_e] for i in range(N)]

    if p_e > 1 or p_e <= 0:
        raise ValueError("Partial Efficiency not in (0,1]")

    DRP = np.zeros([N, N], dtype=bool)
    DRSP = np.zeros([N, N], dtype=bool)

    if crit == 'MON':
        for i, j in cart_range(N):
            DRP[i, j] = (e[i, j] <= expen[i])
            DRSP[i, j] = (e[i, j] < expen[i])

    elif crit == 'FOSD':
        for i, j in cart_range(N):
            DRP[i, j] = any(np.dot(p[i], np.array(y)) <= expen[i]
                            for y in permutations(x[j]))
            DRSP[i, j] = any(np.dot(p[i], np.array(y)) < expen[i]
                             for y in permutations(x[j]))

    else:
        raise ValueError("Unrecognized criterion for revealed preferences")

    # Solve numerical inconsistencies
    if crit == 'MON':
        for i, j in cart_range(N):
            if np.array_equal(x[i], x[j]):
                DRP[i, j] = (p_e == 1)
                DRSP[i, j] = False

    return DRP, DRSP


def indirect_rev_prefs(DRP, DRSP):
    N = DRP.shape[0]

    RP = tran_clos(DRP)
    RSP = np.zeros([N, N], dtype=bool)

    for i, j in cart_range(N):
        for m, n in cart_range(N):
            if RP[i, m] and DRSP[m, n] and RP[n, j]:
                RSP[i, j] = True
                break

    return RP, RSP


def cycles_from_graph(g, max_length=None, max_num=None):
    # G is directed graph
    # max_length is maximum length of cycles
    # max_num is maximum number of cycles. If graph has more cycles than cutoff
    #   then function returns -1, []

    success = True

    # Try to enumerate max_num + 1 simple cycles
    cycles = []
    if max_num is None:
        for c in simple_cycles(g, length_bound=max_length):
            cycles.append(c)

    else:
        for c in islice(simple_cycles(g, length_bound=max_length),
                        max_num + 1):
            cycles.append(c)

        if len(cycles) > max_num:
            success = False
            cycles = []

    return success, cycles


def garp(p, x, p_e=1, crit='MON'):
    # Returns True if dataset (p, x) satisfies GARP, False otherwise
    # p_e is partial efficiency level
    # crit is dominance criterion (MON or FOSD)

    DRP, DRSP = direct_rev_prefs(p, x, p_e, crit)
    RP = tran_clos(DRP)

    GARP = True
    for i, j in cart_range(DRP.shape[0]):
        if RP[i, j] and DRSP[j, i]:
            GARP = False
            break

    return GARP


def tran_clos(e):
    # Receive square boolean matrix e, return transitive closure matrix
    #   e[i, j] = True if there is a path from i to j

    # First convert matrix to dictionary M
    M = {}
    N = e.shape[0]
    for i in range(N):
        M[i] = [j for j in range(N) if e[i, j]]

    # Get transitive closure
    M_tc = tc(M)

    e_tc = np.zeros([N, N], dtype=bool)
    for i, j in cart_range(N):
        if j in M_tc[i]:
            e_tc[i, j] = True

    return e_tc


def to_unique(x, DRP, DRSP):
    # Transform x and revealed preferences to list with unique bundles

    i = 0
    while i < x.shape[0]:
        N = x.shape[0]

        remove = []
        for j in range(i + 1, N):
            if np.array_equal(x[i], x[j]):  # If x[i]=x[j]

                # Add observation j to list of observations to delete
                remove.append(j)

                # x[i] is revealed preferred to x[k] if either x[i] is
                #   revealed preferred to x[k] or if x[j](=x[i]) is revealed
                #   preferred to x[k]
                for k in range(N):
                    DRP[i, k] = (DRP[i, k] or DRP[j, k])
                    DRSP[i, k] = (DRSP[i, k] or DRSP[j, k])

        # Remove observations j such that x^j==x^i:
        #   Remove j^{th} row and column from matrix DRP
        #   (Remove j^{th} observation)
        x = np.delete(x, remove, axis=0)
        DRP = np.delete(DRP, remove, axis=0)
        DRP = np.delete(DRP, remove, axis=1)
        DRSP = np.delete(DRSP, remove, axis=0)
        DRSP = np.delete(DRSP, remove, axis=1)

        i += 1

    return x, DRP, DRSP


def dom_weak(x, y):
    dom = True
    for i in range(x.shape[0]):
        if y[i] > x[i]:
            dom = False
            break

    return dom


def dom_strict(x, y):
    # Returns True if x strictly dominates y, False otherwise
    if np.array_equal(x, y):
        return False

    for i in range(x.shape[0]):
        if y[i] > x[i]:
            return False

    return True


def read_data(file, subsample):
    df = pd.read_csv(f'../data/final_{file}.csv')

    # Set values starting from zero given python 0-index
    df['round'] -= 1

    df = df[df['type'] == subsample]

    ids = list(df['id'].unique())
    p = {}
    x = {}
    for i in ids:
        p[i], x[i] = get_subject_data(df, i)

    return p, x


def get_subject_data(df, i):
    df = df[df['id'] == i]

    if 'z' in df.keys():
        p = df[['p_x', 'p_y', 'p_z']].to_numpy()
        x = df[['x', 'y', 'z']].to_numpy()

    else:
        p = df[['p_x', 'p_y']].to_numpy()
        x = df[['x', 'y']].to_numpy()

    return p, x
