"""
quantumlike_bit_generator.py
A module for generating coupled k-regular graphs.
This module creates two k-regular subgraphs and couples them with flexible coupling specifications.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from k_regular_graph_generator import generate_k_regular_graph, get_adjacency_matrix

def couple_graphs_custom(adj_matrix1, adj_matrix2, coupling_spec):
    """
    Couple two graphs with flexible coupling specification.
    Ensures that off-diagonal blocks are transposes of each other (symmetric coupling).
    
    Args:
        adj_matrix1 (numpy.ndarray): Adjacency matrix of first graph
        adj_matrix2 (numpy.ndarray): Adjacency matrix of second graph
        coupling_spec: Coupling specification, can be:
            - int: Number of connections per node (balanced coupling)
            - 'full': Complete bipartite coupling between subgraphs
            - dict: Custom coupling {node_from_graph1: [nodes_from_graph2], ...}
            - numpy.ndarray: Direct coupling matrix (N x N)
    
    Returns:
        numpy.ndarray: Combined adjacency matrix of the coupled graph
    """
    N = adj_matrix1.shape[0]
    
    # Validate that both matrices are same size
    if adj_matrix2.shape[0] != N:
        raise ValueError("Both adjacency matrices must have the same size")
    
    # Create combined adjacency matrix (2N x 2N)
    combined_matrix = np.zeros((2*N, 2*N), dtype=int)
    combined_matrix[:N, :N] = adj_matrix1
    combined_matrix[N:, N:] = adj_matrix2
    
    # Handle different coupling specifications
    if isinstance(coupling_spec, dict):
        # Custom coupling using dictionary
        coupling_matrix = create_custom_coupling(N, coupling_spec)
    elif isinstance(coupling_spec, np.ndarray):
        # Direct coupling matrix provided
        if coupling_spec.shape != (N, N):
            raise ValueError(f"Coupling matrix must be {N}x{N}, got {coupling_spec.shape}")
        coupling_matrix = coupling_spec.astype(int)
    elif coupling_spec == 'full':
        # Complete bipartite coupling
        coupling_matrix = np.ones((N, N), dtype=int)
    elif isinstance(coupling_spec, int):
        # Balanced coupling with specified number of connections per node
        if coupling_spec > N:
            raise ValueError(f"Coupling parameter ({coupling_spec}) cannot exceed graph size N ({N})")
        coupling_matrix = create_balanced_coupling(N, coupling_spec)
    else:
        raise ValueError("coupling_spec must be int, 'full', dict, or numpy array")
    
    # Apply symmetric coupling
    combined_matrix[:N, N:] = coupling_matrix
    combined_matrix[N:, :N] = coupling_matrix.T
    
    return combined_matrix

def create_custom_coupling(N, coupling_dict):
    """
    Create coupling matrix from dictionary specification.
    
    Args:
        N (int): Size of each subgraph
        coupling_dict (dict): {node_from_graph1: [nodes_from_graph2], ...}
    
    Returns:
        numpy.ndarray: Coupling matrix (N x N)
    """
    coupling_matrix = np.zeros((N, N), dtype=int)
    
    for source_node, target_nodes in coupling_dict.items():
        # Validate source node
        if not (0 <= source_node < N):
            raise ValueError(f"Source node {source_node} is out of range [0, {N-1}]")
        
        # Handle single target or list of targets
        if isinstance(target_nodes, int):
            target_nodes = [target_nodes]
        
        # Validate and set connections
        for target_node in target_nodes:
            if not (0 <= target_node < N):
                raise ValueError(f"Target node {target_node} is out of range [0, {N-1}]")
            coupling_matrix[source_node, target_node] = 1
    
    return coupling_matrix

def create_balanced_coupling(N, l):
    """
    Create balanced coupling where in-degrees are as uniform as possible.
    
    Args:
        N (int): Size of each subgraph
        l (int): Number of connections per node
    
    Returns:
        numpy.ndarray: Coupling matrix (N x N)
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

# Keep original function name for backwards compatibility
def couple_graphs_random(adj_matrix1, adj_matrix2, l):
    """
    Legacy function for backwards compatibility.
    Use couple_graphs_custom() for new applications.
    """
    return couple_graphs_custom(adj_matrix1, adj_matrix2, l)

# Helper functions for common coupling patterns
def create_mirror_coupling(N):
    """
    Create mirror coupling where node i connects to node i (1-to-1 alignment).
    This ensures straight-line connections with no crossing.
    
    Args:
        N (int): Size of each subgraph
    
    Returns:
        dict: Coupling specification for mirror coupling
    """
    coupling_spec = {}
    for i in range(N):
        coupling_spec[i] = [i]
    return coupling_spec

def create_ring_coupling(N):
    """
    Create ring-like coupling where each node connects to its neighbor.
    
    Args:
        N (int): Size of each subgraph
    
    Returns:
        dict: Coupling specification for ring coupling
    """
    coupling_spec = {}
    for i in range(N):
        coupling_spec[i] = [(i + 1) % N]  # Each node connects to next node (circular)
    return coupling_spec

def create_offset_coupling(N, offset=1):
    """
    Create offset coupling where node i connects to node (i+offset) % N.
    
    Args:
        N (int): Size of each subgraph
        offset (int): Offset for connections
    
    Returns:
        dict: Coupling specification for offset coupling
    """
    coupling_spec = {}
    for i in range(N):
        coupling_spec[i] = [(i + offset) % N]
    return coupling_spec

def create_star_coupling(N, center_nodes=None):
    """
    Create star-like coupling where specified center nodes connect to all nodes.
    
    Args:
        N (int): Size of each subgraph
        center_nodes (list): List of center nodes, defaults to [0]
    
    Returns:
        dict: Coupling specification for star coupling
    """
    if center_nodes is None:
        center_nodes = [0]
    
    coupling_spec = {}
    for center in center_nodes:
        if 0 <= center < N:
            coupling_spec[center] = list(range(N))
    return coupling_spec


def generate_ql_bit(N, k, coupling_spec, num_candidate_graphs=10):
    """
    Generate two coupled k-regular graphs with flexible coupling.
    
    Args:
        N (int): Number of nodes in each subgraph
        k (int): Degree of each node within subgraph (k-regular)
        coupling_spec: Coupling specification, can be:
            - int: Number of connections per node (balanced coupling)
            - 'full': Complete bipartite coupling between subgraphs
            - dict: Custom coupling {node_from_graph1: [nodes_from_graph2], ...}
            - numpy.ndarray: Direct coupling matrix (N x N)
        num_candidate_graphs (int): Number of candidate graphs to generate before selection
    
    Returns:
        tuple: (coupled_adjacency_matrix, graph_info)
            - coupled_adjacency_matrix (numpy.ndarray): 2N x 2N adjacency matrix
            - graph_info (dict): Information about the generated graphs
    """
    # Validation for integer coupling (backwards compatibility)
    if isinstance(coupling_spec, int):
        if coupling_spec > N:
            raise ValueError(f"Coupling parameter ({coupling_spec}) cannot exceed graph size N ({N})")
    
    if k >= N:
        raise ValueError(f"Degree k ({k}) must be less than graph size N ({N})")
    if (N * k) % 2 != 0:
        raise ValueError(f"N*k ({N*k}) must be even for valid k-regular graphs")
    
    # Generate two k-regular graphs directly (more efficient)
    G1 = generate_k_regular_graph(N, k)
    G2 = generate_k_regular_graph(N, k)
    
    adj_matrix1 = get_adjacency_matrix(G1)
    adj_matrix2 = get_adjacency_matrix(G2)
    
    # Couple the graphs using the new flexible method
    coupled_matrix = couple_graphs_custom(adj_matrix1, adj_matrix2, coupling_spec)
    
    # Calculate coupling statistics based on coupling type
    coupling_matrix = coupled_matrix[:N, N:]  # Extract coupling block
    total_coupling_edges = np.sum(coupling_matrix)
    
    # Determine coupling degree info based on type
    if isinstance(coupling_spec, int):
        coupling_degree = coupling_spec
        coupling_type = "balanced"
    elif coupling_spec == 'full':
        coupling_degree = N
        coupling_type = "complete"
    elif isinstance(coupling_spec, dict):
        coupling_degree = "custom"
        coupling_type = "custom"
    elif isinstance(coupling_spec, np.ndarray):
        coupling_degree = "matrix"
        coupling_type = "matrix"
    else:
        coupling_degree = "unknown"
        coupling_type = "unknown"
    
    # Prepare graph information
    graph_info = {
        'subgraph_size': N,
        'subgraph_degree': k,
        'coupling_degree': coupling_degree,
        'coupling_type': coupling_type,
        'coupling_spec': coupling_spec,  # Store original specification
        'total_size': 2 * N,
        'total_edges': np.sum(coupled_matrix) // 2,
        'coupling_edges': total_coupling_edges,
        'internal_edges_per_subgraph': np.sum(adj_matrix1) // 2
    }
    
    return coupled_matrix, graph_info


def visualize_ql_bit(coupled_matrix, graph_info, show_plot=True, save_path=None):
    """
    Clean separated layout visualization for coupled graphs.
    Updated to handle flexible coupling specifications.
    
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
    coupling_degree = graph_info['coupling_degree']
    coupling_type = graph_info['coupling_type']
    total_coupling_edges = graph_info['coupling_edges']
    
    # Create NetworkX graph from coupled matrix
    coupled_graph = nx.from_numpy_array(coupled_matrix)
    
    # Scaling parameters based on graph size
    if N <= 6:
        fig_width, fig_height = 12, 6
        node_size = 600
        font_size = 12
        internal_edge_width = 6
        coupling_edge_width = 2
        separation_distance = 2.5
    elif N <= 15:
        fig_width, fig_height = 15, 8
        node_size = 400
        font_size = 10
        internal_edge_width = 4
        coupling_edge_width = 2
        separation_distance = 3.0
    else:  # Large graphs
        fig_width, fig_height = 18, 10
        node_size = 300
        font_size = 8
        internal_edge_width = 3
        coupling_edge_width = 2
        separation_distance = 3.5
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Define node lists for each subgraph
    subgraph1_nodes = list(range(N))
    subgraph2_nodes = list(range(N, 2*N))
    
    # Create clean separated positions - this is the approach that works well
    pos = {}
    
    # Spring layout for subgraph 1 (positioned on left)
    subgraph1 = coupled_graph.subgraph(subgraph1_nodes)
    pos1 = nx.spring_layout(subgraph1, seed=42)
    for node, (x, y) in pos1.items():
        pos[node] = (x - separation_distance, y)
    
    # Spring layout for subgraph 2 (positioned on right)
    subgraph2 = coupled_graph.subgraph(subgraph2_nodes)
    pos2 = nx.spring_layout(subgraph2, seed=43)
    for node, (x, y) in pos2.items():
        pos[node] = (x + separation_distance, y)
    
    # Define edge lists
    # Internal edges for subgraph 1
    subgraph1_edges = [(u, v) for u, v in coupled_graph.edges() 
                       if u in subgraph1_nodes and v in subgraph1_nodes]
    
    # Internal edges for subgraph 2  
    subgraph2_edges = [(u, v) for u, v in coupled_graph.edges()
                       if u in subgraph2_nodes and v in subgraph2_nodes]
    
    # Coupling edges (between subgraphs)
    coupling_edges = [(u, v) for u, v in coupled_graph.edges()
                     if (u in subgraph1_nodes and v in subgraph2_nodes) or 
                        (u in subgraph2_nodes and v in subgraph1_nodes)]
    
    # Draw edges with different styles and colors
    # 1. Draw all edges first (thin, light gray background for context)
    nx.draw_networkx_edges(coupled_graph, pos, width=0.5, alpha=0.3, 
                          edge_color='lightgray', ax=ax)
    
    # 2. Draw internal edges for subgraph 1 (thick, red)
    if subgraph1_edges:
        nx.draw_networkx_edges(coupled_graph, pos, edgelist=subgraph1_edges,
                              width=internal_edge_width, alpha=0.6, 
                              edge_color='tab:red', ax=ax)
    
    # 3. Draw internal edges for subgraph 2 (thick, blue)
    if subgraph2_edges:
        nx.draw_networkx_edges(coupled_graph, pos, edgelist=subgraph2_edges,
                              width=internal_edge_width, alpha=0.6, 
                              edge_color='tab:blue', ax=ax)
    
    # 4. Draw coupling edges (medium width, black - fixed color mismatch)
    if coupling_edges:
        nx.draw_networkx_edges(coupled_graph, pos, edgelist=coupling_edges,
                              width=coupling_edge_width, alpha=0.8, 
                              edge_color='black', ax=ax)
    
    # Draw nodes with consistent styling (no special anchor highlighting)
    node_options = {"edgecolors": "tab:gray", "node_size": node_size, "alpha": 0.9}
    
    # Draw subgraph 1 nodes (red)
    nx.draw_networkx_nodes(coupled_graph, pos, nodelist=subgraph1_nodes, 
                          node_color="tab:red", **node_options, ax=ax)
    
    # Draw subgraph 2 nodes (blue)
    nx.draw_networkx_nodes(coupled_graph, pos, nodelist=subgraph2_nodes, 
                          node_color="tab:blue", **node_options, ax=ax)
    
    # Draw labels with mathematical notation
    labels = {}
    for i in range(N):
        labels[i] = f'$a_{{{i}}}$'  # Mathematical notation for subgraph 1
    for i in range(N, 2*N):
        labels[i] = f'$b_{{{i-N}}}$'  # Mathematical notation for subgraph 2
    
    nx.draw_networkx_labels(coupled_graph, pos, labels, 
                           font_size=font_size, font_color="white", 
                           font_weight='bold', ax=ax)
    
    # Create adaptive title based on coupling type
    if coupling_type == "balanced":
        title = f"Coupled Graph: Two {k}-Regular Subgraphs ({N} nodes each) with {coupling_degree} Coupling Edges per Node"
    elif coupling_type == "complete":
        title = f"Coupled Graph: Two {k}-Regular Subgraphs ({N} nodes each) with Complete Coupling"
    elif coupling_type == "custom":
        title = f"Coupled Graph: Two {k}-Regular Subgraphs ({N} nodes each) with Custom Coupling ({total_coupling_edges} edges)"
    elif coupling_type == "matrix":
        title = f"Coupled Graph: Two {k}-Regular Subgraphs ({N} nodes each) with Matrix Coupling ({total_coupling_edges} edges)"
    else:
        title = f"Coupled Graph: Two {k}-Regular Subgraphs ({N} nodes each) with {total_coupling_edges} Coupling Edges"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Clean legend
    from matplotlib.patches import Patch
    import matplotlib.lines as mlines
    
    legend_elements = [
        Patch(facecolor='tab:red', edgecolor='tab:gray', label='Subgraph 1 nodes'),
        Patch(facecolor='tab:blue', edgecolor='tab:gray', label='Subgraph 2 nodes'),
        mlines.Line2D([], [], color='tab:red', linewidth=internal_edge_width, 
                     label='Subgraph 1 internal edges'),
        mlines.Line2D([], [], color='tab:blue', linewidth=internal_edge_width, 
                     label='Subgraph 2 internal edges'),
        mlines.Line2D([], [], color='black', linewidth=coupling_edge_width, 
                     label='Coupling edges')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.15, 1), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig