import numpy as np
import pandas as pd
from compute_pcoeffs import *

import numpy as np

def generate_dummy_data(num_nodes, num_communities):
    # Generate random connectivity matrix W
    W = np.random.rand(num_nodes, num_nodes)
    # Ensure symmetry by averaging with transpose
    W = 0.5 * (W + W.T)
    # Generate random community assignments Ci
    Ci = np.random.randint(1, num_communities + 1, size=num_nodes)
    return W, Ci


# Case 1: Non-participation scenario
num_nodes = 5
num_communities = 1
W_case1, Ci_case1 = generate_dummy_data(num_nodes, num_communities)

PC_norm_case1, _, _, _ = participation_coef_norm(W_case1, Ci_case1, n_iter=10)
print("Case 1 - Non-participation scenario:")
print("Participation Coefficients:", PC_norm_case1)

# Case 2: Full participation scenario
num_nodes = 5
num_communities = 5
W_case2, Ci_case2 = generate_dummy_data(num_nodes, num_communities)

PC_norm_case2, _, _, _ = participation_coef_norm(W_case2, Ci_case2, n_iter=10)
print("\nCase 2 - Full participation scenario:")
print("Participation Coefficients:", PC_norm_case2)

# Case 3: Intermediate participation scenario
num_nodes = 20
num_communities = 5
W_case3, Ci_case3 = generate_dummy_data(num_nodes, num_communities)

PC_norm_case3, _, _, _ = participation_coef_norm(W_case3, Ci_case3, n_iter=10)
print("\nCase 3 - Intermediate participation scenario:")
print("Participation Coefficients:", PC_norm_case3)
