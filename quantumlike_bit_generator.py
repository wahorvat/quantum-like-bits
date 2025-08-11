"""
quantumlike_bit_generator.py

A module for generating coupled k-regular graphs.
This module creates two k-regular subgraphs and couples them with m connections per node.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from k_regular_graph_generator import generate_k_regular_graph, get_adjacency_matrix


def couple_graphs(adj_matrix1, adj_matrix2, l):
    """
    Couple two graphs by adding l connections per node between the subgraphs using balanced coupling.
    Ensures that off-diagonal blocks are transposes of each other.
    Args:
        adj_matrix1 (numpy.ndarray): Adjacency matrix of first graph
        adj_matrix2 (numpy.ndarray): Adjacency matrix of second graph
        l (int or 'full'): Number of connections per node to the other subgraph, or 'full' for complete coupling
    Returns:
        numpy.ndarray: Combined adjacency matrix of the coupled graph
    """
    N = adj_matrix1.shape[0]
    
    # Create combined adjacency matrix (2N x 2N)
    combined_matrix = np.zeros((2*N, 2*N), dtype=int)
    combined_matrix[:N, :N] = adj_matrix1
    combined_matrix[N:, N:] = adj_matrix2
    
    if l == 'full':
        coupling_matrix = np.ones((N, N), dtype=int)
    else:
        if l > N:
            raise ValueError(f"Coupling parameter l ({l}) cannot exceed graph size N ({N})")
        coupling_matrix = create_balanced_coupling(N, l)
    
    combined_matrix[:N, N:] = coupling_matrix
    combined_matrix[N:, :N] = coupling_matrix.T
    return combined_matrix

def create_balanced_coupling(N, l):
    """
    Create balanced coupling where in-degrees are as uniform as possible.
    """
    coupling_matrix = np.zeros((N, N), dtype=int)
    target_in_degree = l
    in_degrees = np.zeros(N, dtype=int)
    
    for i in range(N):
        available_targets = [j for j in range(N) if in_degrees[j] < target_in_degree]
        connections_made = 0
        
        while connections_made < l and available_targets:
            j = min(available_targets, key=lambda x: in_degrees[x])
            coupling_matrix[i, j] = 1
            in_degrees[j] += 1
            connections_made += 1
            
            if in_degrees[j] >= target_in_degree:
                available_targets.remove(j)
    
    return coupling_matrix



def generate_ql_bit(N, k, l, num_candidate_graphs=10):
    """
    Generate two coupled k-regular graphs.
    
    Args:
        N (int): Number of nodes in each subgraph
        k (int): Degree of each node within subgraph (k-regular)
        l (int): Number of coupling connections per node between subgraphs
        num_candidate_graphs (int): Number of candidate graphs to generate before selection
    
    Returns:
        tuple: (coupled_adjacency_matrix, graph_info)
            - coupled_adjacency_matrix (numpy.ndarray): 2N x 2N adjacency matrix
            - graph_info (dict): Information about the generated graphs
    """
    # Validation
    if l > N:
        raise ValueError(f"Coupling parameter l ({l}) cannot exceed graph size N ({N})")
    if k >= N:
        raise ValueError(f"Degree k ({k}) must be less than graph size N ({N})")
    if (N * k) % 2 != 0:
        raise ValueError(f"N*k ({N*k}) must be even for valid k-regular graphs")
    
    # Generate two k-regular graphs directly (more efficient)
    G1 = generate_k_regular_graph(N, k)
    G2 = generate_k_regular_graph(N, k)
    
    adj_matrix1 = get_adjacency_matrix(G1)
    adj_matrix2 = get_adjacency_matrix(G2)
    
    # Couple the graphs with correct block structure
    coupled_matrix = couple_graphs(adj_matrix1, adj_matrix2, l)
    
    # Prepare graph information
    graph_info = {
        'subgraph_size': N,
        'subgraph_degree': k,
        'coupling_degree': l,
        'total_size': 2 * N,
        'total_degree_per_node': k + l,
        'total_edges': np.sum(coupled_matrix) // 2,
        'coupling_edges': N * l
    }
    
    return coupled_matrix, graph_info


def visualize_ql_bit(coupled_matrix, graph_info, show_plot=True, save_path=None):
    """
    Visualize the coupled graph.
    
    Args:
        coupled_matrix (numpy.ndarray): Adjacency matrix of coupled graph
        graph_info (dict): Information about the graph
        show_plot (bool): Whether to display the plot
        save_path (str): Path to save the plot (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    N = graph_info['subgraph_size']
    k = graph_info['subgraph_degree']
    l = graph_info['coupling_degree']
    
    # Create NetworkX graph from coupled matrix
    coupled_graph = nx.from_numpy_array(coupled_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Create layout that separates the two subgraphs visually
    pos = {}
    subgraph1_nodes = list(range(N))
    subgraph2_nodes = list(range(N, 2*N))
    
    # Position subgraph 1 on the left
    subgraph1 = coupled_graph.subgraph(subgraph1_nodes)
    pos1 = nx.spring_layout(subgraph1, seed=42)
    for node, (x, y) in pos1.items():
        pos[node] = (x - 2, y)
    
    # Position subgraph 2 on the right
    subgraph2 = coupled_graph.subgraph(subgraph2_nodes)
    pos2 = nx.spring_layout(subgraph2, seed=43)
    for node, (x, y) in pos2.items():
        pos[node] = (x + 2, y)
    
    # Draw the coupled graph with different colors for each subgraph
    node_colors = ['lightblue' if node < N else 'lightcoral' for node in coupled_graph.nodes()]
    
    # Draw all edges
    nx.draw_networkx_edges(coupled_graph, pos, alpha=0.5, width=1, edge_color='gray', ax=ax)
    
    # Highlight coupling edges (between subgraphs)
    coupling_edges = [(u, v) for u, v in coupled_graph.edges() 
                      if (u < N and v >= N) or (u >= N and v < N)]
    nx.draw_networkx_edges(coupled_graph, pos, edgelist=coupling_edges, 
                          width=2, edge_color='red', alpha=0.8, ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(coupled_graph, pos, node_color=node_colors, 
                          node_size=400, alpha=0.9, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(coupled_graph, pos, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title(f"Coupled Graph: Two {k}-Regular Subgraphs ({N} nodes each) \n with l={l} Inter-connections", 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightblue', label='Subgraph 1'),
                      Patch(facecolor='lightcoral', label='Subgraph 2'),
                      plt.Line2D([0], [0], color='red', lw=2, label='Coupling edges')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig