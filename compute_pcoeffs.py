import numpy as np
from bct import null_model_und_sign, randmio_dir
import scipy.sparse as sp

def participation_coef(W, Ci):
    n = W.shape[0]  # number of vertices
    Ko = np.array(W.sum(axis=1)).flatten().astype(float)  # (out) degree
    Gc = np.dot((W != 0), np.diag(Ci))  # neighbor community affiliation
    Gc = sp.csr_matrix(Gc)
    P = np.zeros((n))  # community-specific neighbors

    # Assuming Gc is a sparse matrix
    for i in range(1, int(np.max(Ci)) + 1):
        P = P + (np.array((sp.csr_matrix(W).multiply(Gc == i).astype(int)).sum(axis=1)).flatten() / Ko)**2
    P = 1 - P
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0    
    return P


def participation_coef_norm(W, Ci, n_iter=10, par_comp=0):
    import time
    start_time = time.time()
    
    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out)degree
    #nonzero_indices = np.nonzero(W)

# Multiply the non-zero values of W with the diagonal elements of Ci
    #Gc = np.zeros_like(W)  # Initialize Gc with zeros
    #Gc[nonzero_indices] = W[nonzero_indices] * np.diag(Ci)
    Gc = W  @ np.diag(Ci)  # neighbor community affiliation [ Do we need to do np.matmul((W != 0), np.diag(Ci)) ?]
    #nonzero_indices = np.nonzero(W)

# Reshape the indices to match the shape of the diagonal matrix Ci
    #nonzero_indices_2d = (nonzero_indices[0], nonzero_indices[1])

    Kc2 = np.zeros(n)
    within_mod_k = np.zeros(n)
    for i in range(1, max(Ci)+1):
        Kc2 += (np.sum(W @ (Gc == i), axis=1) ** 2)  # squared intra-modular degree
        within_mod_k[Ci == i] = np.sum(W[Ci == i, :][:, Ci == i], axis=1)  # within-module degree
    
    between_mod_k = Ko - within_mod_k  # [network-wide degree - within-module degree = between-module degree]
    PC = 1 - Kc2 / (Ko ** 2)  # calculate participation coefficient
    PC[Ko == 0] = 0  # PC = 0 for nodes with no (out)neighbors
    
    Kc2_rnd = np.zeros((n, n_iter))  # initialize randomized intra-modular degree array
    
    if par_comp == 0:  # no parallel computing
        for ii in range(n_iter):  # number of randomizations
            W_rnd = null_model_und_sign(W, bin_swaps=5)  # randomize each undirected network five times
            W_rnd = np.round(W_rnd[0]).astype(int)  # randomize each undirected network five times, preserving degree distribution of original matrix
            Gc_rnd = (W_rnd != 0) * np.diag(Ci)  # neighbor community affiliation
            Kc2_rnd_loop = np.zeros(n)  # initialize randomized intramodular degree vector - for current iteration only
            
            for j in range(1, max(Ci)+1):
                Kc2_rnd_loop += (np.sum(W_rnd * (Gc_rnd == j), axis=1) ** 2)
            Kc2_rnd[:, ii] = Kc2_rnd_loop
    
    PC_norm = np.mean(1 - Kc2_rnd / (Ko[:, None] ** 2), axis=1)
    PC_residual = PC - PC_norm
    
    elapsed_time = time.time() - start_time
    print(f"\n\t - Finished Normalized Participation Coefficient with {n_iter} randomizations in {elapsed_time:.2f} seconds.")
    
    return PC_norm, PC_residual, PC, between_mod_k

# Example usage:
#num_nodes = 20  # Number of nodes
#num_communities = 5  # Number of communities
# W = thresholded_matrix
# #W = w.to_numpy()

# # Load the community assignments for each node
# df = pd.read_excel("/Users/subhasriviswanathan/Downloads/GraphVar_2.03a/ROI_templates/seitzman_atlas_final.xlsx", header=None)
# Ci= df.iloc[:,8].values
# Ci.shape
# #W, Ci = generate_dummy_data(num_nodes, num_communities)

# PC_norm, PC_residual, PC, between_mod_k = participation_coef_norm(W, Ci, n_iter=10)


# create new conda env
# conda create -n myenv python=3.8