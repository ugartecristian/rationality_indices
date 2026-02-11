import numpy as np
from tarjan.tc import tc
from functions.utils_cu import direct_rev_prefs, cart_range

# Solve CCEI


def CCEI(p, x):
    # Get list of candidates for CCEI
    E = np.dot(p, x.T)
    cand = np.unique(E)
    cand = cand[cand <= 1]

    # Find CCEI. cand vector sorts candidates from smaller to bigger, so start
    # at the end
    i = cand.shape[0] - 1

    while True:

        DRP, DRSP = direct_rev_prefs(p, x, p_e=cand[i])

        if is_acyclic(DRSP):
            ccei = cand[i]
            break
        else:
            i -= 1

    return ccei


def is_acyclic(X):
    # get boolean square matrix X. Return True if X is acyclic, False otherwise

    # First convert matrix to dictionary M
    M = {}
    N = X.shape[0]
    for i in range(N):
        M[i] = [j for j in range(N) if X[i, j]]

    # Get transitive closure
    M_tc = tc(M)

    # Check if X is acyclic
    for i, j in cart_range(N):
        if X[i, j] and i in M_tc[j]:
            return False

    return True
