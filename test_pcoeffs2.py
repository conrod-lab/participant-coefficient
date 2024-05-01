import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from compute_pcoeffs import *


def generate_dummy_data(num_communities, max_community_node, min_community_node, interconnectivity):
    # Generate community sizes
    community_sizes = np.random.randint(min_community_node, max_community_node + 1, size=num_communities)
    
    # Generate community assignments Ci
    community_assignments = []
    for i, size in enumerate(community_sizes):
        community_assignments.extend([i+1] * size)
    Ci = np.array(community_assignments)
    
    # Generate random connectivity matrix W within communities
    W_within_communities = np.zeros((len(Ci), len(Ci)))
    for community_id in range(1, num_communities + 1):
        community_nodes = np.where(Ci == community_id)[0]
        for i in range(len(community_nodes)):
            for j in range(i + 1, len(community_nodes)):
                W_within_communities[community_nodes[i], community_nodes[j]] = 1
                W_within_communities[community_nodes[j], community_nodes[i]] = 1
    
    # Connect the selected node to two other nodes from each other community
    selected_node = np.random.choice(np.where(Ci == 1)[0])  # Select a node from the first community
    #other_nodes = np.where(Ci != 1)[0]  # Nodes from other communities
    for community_id in range(2, num_communities + 1):
        connections = np.random.choice(np.where(Ci == community_id)[0], size=2, replace=False)
        for node in connections:
            W_within_communities[selected_node, node] = 1
            W_within_communities[node, selected_node] = 1
    
    # Drop an edge within the first community
    first_community_nodes = np.where(Ci == 2)[0]
    node1, node2 = np.random.choice(first_community_nodes, size=2, replace=False)
    W_within_communities[node1, node2] = 0
    W_within_communities[node2, node1] = 0    

    # Combine within community connections
    W = W_within_communities
    
    return W, Ci, selected_node


def plot_network_with_participation(G, PC_norm, selected_node):
    # Draw the network
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
    plt.title('Brain Network with Participation Coefficients')
    
    # Add participation coefficient as node label for the selected node
    selected_PC = PC_norm[selected_node]
    nx.draw_networkx_labels(G, pos, labels={selected_node: f'{selected_node}\nPC={selected_PC:.2f}'}, font_size=8)
    
    plt.show()

# Case 1: Non-participation scenario
# Generate dummy data
num_communities = 3
max_community_node = 3
min_community_node = 3
interconnectivity = 0.1  # Probability of being connected to a community
W, Ci, selected_node = generate_dummy_data(num_communities, max_community_node, min_community_node, interconnectivity)

PC = participation_coef(W, Ci)

PC_norm_case1, _, _, _ = participation_coef_norm(W, Ci, n_iter=10)
print("Case 1 - Non-participation scenario:")
print("Participation Coefficients:", PC_norm_case1)

# Create NetworkX graph
G_case1 = nx.from_numpy_array(W)

# Plot network with participation coefficients
plot_network_with_participation(G_case1, PC,selected_node)

# Case 2: Full participation scenario
num_communities = 3
max_community_node = 6
min_community_node = 3
interconnectivity = 0.1  # Probability of being connected to a community
W, Ci, selected_node = generate_dummy_data(num_communities, max_community_node, min_community_node, interconnectivity)

PC = participation_coef(W, Ci)

PC_norm_case2, _, _, _ = participation_coef_norm(W, Ci, n_iter=10)
print("\nCase 2 - Full participation scenario:")
print("Participation Coefficients:", PC_norm_case2)

# Create NetworkX graph
G_case1 = nx.from_numpy_array(W)

# Plot network with participation coefficients
plot_network_with_participation(G_case1, PC,selected_node)

# Case 3: Intermediate participation scenario
num_nodes = 20
num_communities = 5
W_case3, Ci_case3 = generate_dummy_data(num_nodes, num_communities)

PC_norm_case3, _, _, _ = participation_coef_norm(W_case3, Ci_case3, n_iter=10)
print("\nCase 3 - Intermediate participation scenario:")
print("Participation Coefficients:", PC_norm_case3)

# Create NetworkX graph
G_case3 = nx.from_numpy_array(W_case3)

# Plot network with participation coefficients
plot_network_with_participation(G_case3, PC_norm_case3)
