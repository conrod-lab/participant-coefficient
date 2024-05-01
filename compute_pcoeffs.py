import numpy as np
from bct import null_model_und_sign, participation_coef

#PC_norm = 1 - \sqrt(B_0 \sum_{m \in M} ( \frac{(k_{i}(m) - k_{i,rand}(m)}{K_i})^2 )

def participation_coef_norm(W, Ci, n_iter=10, par_comp=0):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Parameters
    ----------
    W : NxN np.ndarray
        binary/weighted directed/undirected connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector

    Returns
    -------
    P : Nx1 np.ndarray
        participation coefficient
    '''

    _, Ci = np.unique(Ci, return_inverse=True)
    Ci += 1
    B0 = 0.5  # for contraining PCnorm
    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(Ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    Kc2_rnd = np.zeros((n, n_iter))  # initialize randomized intra-modular degree array
    
    if par_comp == 0:  # no parallel computing
        for ii in range(n_iter):  # number of randomizations
            W_rnd = null_model_und_sign(W, bin_swaps=5)  # randomize each undirected network
            W_rnd = np.round(W_rnd[0]).astype(int)  # round to get binary matrix
            Gc_rnd = np.dot((W_rnd != 0), np.diag(Ci))   # neighbor community affiliation
            
            Kc2_rnd_loop = np.zeros(n)  # initialize randomized intramodular degree vector - for current iteration only
            for j in range(1, int(np.max(Ci)) + 1):
                Kc2_rnd_loop += np.sum(W_rnd * (Gc_rnd == j), axis=1) ** 2
            Kc2_rnd[:, ii] = Kc2_rnd_loop

    for i in range(1, int(np.max(Ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    #P = np.ones((n,)) - np.sqrt(B0*(Kc2_rnd * par_comp + Kc2) / np.square(Ko))
    PC = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    PC[np.where(np.logical_not(Ko))] = 0
    PC_norm = np.mean(1 - Kc2_rnd / (Ko[:, None] ** 2), axis=1)
    PC_residual = PC - PC_norm
    # P=0 for nodes with no (out) neighbors

    return PC_norm, PC_residual, PC, Ko
