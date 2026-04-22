import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from k_regular_graph_generator import generate_k_regular_graph, get_adjacency_matrix

def generate_quantum_like_bit(N1, k1, N2, k2, coupling_spec):
    """
    Generate a quantum-like bit: two differently-sized k-regular graphs coupled with flexible coupling.
    
    Args:
        N1 (int): Number of nodes in first k1-regular subgraph
        k1 (int): Degree of each node in first subgraph (k1-regular)
        N2 (int): Number of nodes in second k2-regular subgraph  
        k2 (int): Degree of each node in second subgraph (k2-regular)
        coupling_spec: Coupling specification, can be:
            - int: Number of connections per node (balanced coupling)
            - 'full': Complete bipartite coupling between subgraphs
            - dict: Custom coupling {node_from_graph1: [nodes_from_graph2], ...}
            - numpy.ndarray: Direct coupling matrix (N1 x N2)
    
    Returns:
        tuple: (coupled_adjacency_matrix, graph_info)
            - coupled_adjacency_matrix (numpy.ndarray): (N1+N2) x (N1+N2) adjacency matrix
            - graph_info (dict): Information about the generated graphs
    """
    # Input validation for first graph
    if k1 >= N1:
        raise ValueError(f"Degree k1 ({k1}) must be less than graph size N1 ({N1})")
    if (N1 * k1) % 2 != 0:
        raise ValueError(f"N1*k1 ({N1*k1}) must be even for valid k1-regular graphs")
    
    # Input validation for second graph
    if k2 >= N2:
        raise ValueError(f"Degree k2 ({k2}) must be less than graph size N2 ({N2})")
    if (N2 * k2) % 2 != 0:
        raise ValueError(f"N2*k2 ({N2*k2}) must be even for valid k2-regular graphs")
    
    # Input validation for coupling
    if isinstance(coupling_spec, int):
        if coupling_spec > min(N1, N2):
            raise ValueError(f"Coupling parameter ({coupling_spec}) cannot exceed min(N1, N2) ({min(N1, N2)})")
    
    # Generate two k-regular graphs with different sizes
    G1 = generate_k_regular_graph(N1, k1)
    G2 = generate_k_regular_graph(N2, k2)
    
    # Get adjacency matrices
    adj_matrix1 = get_adjacency_matrix(G1)
    adj_matrix2 = get_adjacency_matrix(G2)
    
    # Couple the graphs using flexible coupling
    combined_matrix = couple_graphs_custom(adj_matrix1, adj_matrix2, coupling_spec)
    
    # Calculate coupling statistics
    coupling_matrix = combined_matrix[:N1, N1:]  # Extract coupling block
    total_coupling_edges = np.sum(coupling_matrix)
    
    # Determine coupling degree info based on type
    if isinstance(coupling_spec, int):
        coupling_degree = coupling_spec
        coupling_type = "balanced"
    elif coupling_spec == 'full':
        coupling_degree = min(N1, N2)  # Each node connects to all nodes in other graph
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
        'subgraph1_size': N1,
        'subgraph1_degree': k1,
        'subgraph2_size': N2,
        'subgraph2_degree': k2,
        'coupling_degree': coupling_degree,
        'coupling_type': coupling_type,
        'coupling_spec': coupling_spec,
        'total_size': N1 + N2,
        'total_edges': np.sum(combined_matrix) // 2,
        'coupling_edges': total_coupling_edges,
        'internal_edges_subgraph1': np.sum(adj_matrix1) // 2,
        'internal_edges_subgraph2': np.sum(adj_matrix2) // 2
    }
    
    return combined_matrix, graph_info


def couple_graphs_custom(adj_matrix1, adj_matrix2, coupling_spec):
    """
    Couple two graphs (potentially different sizes) with flexible coupling specification.
    Ensures that off-diagonal blocks are transposes of each other (symmetric coupling).
    
    Args:
        adj_matrix1 (numpy.ndarray): Adjacency matrix of first graph (N1 x N1)
        adj_matrix2 (numpy.ndarray): Adjacency matrix of second graph (N2 x N2)
        coupling_spec: Coupling specification, can be:
            - int: Number of connections per node (balanced coupling)
            - 'full': Complete bipartite coupling between subgraphs
            - dict: Custom coupling {node_from_graph1: [nodes_from_graph2], ...}
            - numpy.ndarray: Direct coupling matrix (N1 x N2)
    
    Returns:
        numpy.ndarray: Combined adjacency matrix of the coupled graph
    """
    N1 = adj_matrix1.shape[0]
    N2 = adj_matrix2.shape[0]
    
    # Create combined adjacency matrix ((N1+N2) x (N1+N2))
    combined_matrix = np.zeros((N1+N2, N1+N2), dtype=int)
    combined_matrix[:N1, :N1] = adj_matrix1
    combined_matrix[N1:, N1:] = adj_matrix2
    
    # Handle different coupling specifications
    if isinstance(coupling_spec, dict):
        # Custom coupling using dictionary
        coupling_matrix = create_custom_coupling(N1, N2, coupling_spec)
    elif isinstance(coupling_spec, np.ndarray):
        # Direct coupling matrix provided
        if coupling_spec.shape != (N1, N2):
            raise ValueError(f"Coupling matrix must be {N1}x{N2}, got {coupling_spec.shape}")
        coupling_matrix = coupling_spec.astype(int)
    elif coupling_spec == 'full':
        # Complete bipartite coupling
        coupling_matrix = np.ones((N1, N2), dtype=int)
    elif isinstance(coupling_spec, int):
        # Balanced coupling with specified number of connections per node
        if coupling_spec > min(N1, N2):
            raise ValueError(f"Coupling parameter ({coupling_spec}) cannot exceed min(N1, N2) ({min(N1, N2)})")
        coupling_matrix = create_balanced_coupling(N1, N2, coupling_spec)
    else:
        raise ValueError("coupling_spec must be int, 'full', dict, or numpy array")
    
    # Apply symmetric coupling
    combined_matrix[:N1, N1:] = coupling_matrix
    combined_matrix[N1:, :N1] = coupling_matrix.T
    
    return combined_matrix


def create_custom_coupling(N1, N2, coupling_dict):
    """
    Create coupling matrix from dictionary specification for different-sized graphs.
    
    Args:
        N1 (int): Size of first subgraph
        N2 (int): Size of second subgraph
        coupling_dict (dict): {node_from_graph1: [nodes_from_graph2], ...}
    
    Returns:
        numpy.ndarray: Coupling matrix (N1 x N2)
    """
    coupling_matrix = np.zeros((N1, N2), dtype=int)
    
    for source_node, target_nodes in coupling_dict.items():
        # Validate source node
        if not (0 <= source_node < N1):
            raise ValueError(f"Source node {source_node} is out of range [0, {N1-1}]")
        
        # Handle single target or list of targets
        if isinstance(target_nodes, int):
            target_nodes = [target_nodes]
        
        # Validate and set connections
        for target_node in target_nodes:
            if not (0 <= target_node < N2):
                raise ValueError(f"Target node {target_node} is out of range [0, {N2-1}]")
            coupling_matrix[source_node, target_node] = 1
    
    return coupling_matrix


def create_balanced_coupling_symmetric(N1, N2, l):
    """
    Create balanced coupling between different sized graphs where coupling degrees are matched.
    Creates symmetric coupling so that total degrees are balanced between subgraphs.
    
    Args:
        N1 (int): Size of first subgraph
        N2 (int): Size of second subgraph  
        l (int): Target number of coupling connections per node
        
    Returns:
        numpy.ndarray: Symmetric coupling matrix (N1 x N2)
    """
    coupling_matrix = np.zeros((N1, N2), dtype=int)
    
    # Track degrees for both subgraphs
    out_degrees_1 = np.zeros(N1, dtype=int)  # Coupling out-degrees for subgraph 1
    in_degrees_2 = np.zeros(N2, dtype=int)   # Coupling in-degrees for subgraph 2
    
    # Calculate target total coupling edges for balance
    # We want roughly equal coupling degrees for both subgraphs
    if N1 <= N2:
        # Each node in smaller subgraph gets l connections
        target_connections_per_node_1 = l
        target_total_edges = N1 * l
        target_connections_per_node_2 = target_total_edges // N2
    else:
        # Each node in smaller subgraph gets l connections  
        target_connections_per_node_2 = l
        target_total_edges = N2 * l
        target_connections_per_node_1 = target_total_edges // N1
    
    # Phase 1: Create initial asymmetric coupling
    for i in range(N1):
        available_targets = list(range(N2))
        connections_made = 0
        
        while connections_made < target_connections_per_node_1 and available_targets:
            # Choose target with minimum in-degree
            j = min(available_targets, key=lambda x: in_degrees_2[x])
            coupling_matrix[i, j] = 1
            out_degrees_1[i] += 1
            in_degrees_2[j] += 1
            connections_made += 1
            
            # Remove target if it reaches reasonable threshold
            if in_degrees_2[j] >= (target_total_edges // N2) + 1:
                available_targets.remove(j)
    
    # Phase 2: Add reverse connections to balance degrees
    # Calculate how many more connections subgraph 2 nodes need
    current_coupling_degrees_2 = in_degrees_2.copy()  # These become out-degrees after symmetrization
    target_out_degrees_2 = target_connections_per_node_2
    
    for j in range(N2):
        needed_connections = max(0, target_out_degrees_2 - current_coupling_degrees_2[j])
        
        if needed_connections > 0:
            # Find nodes in subgraph 1 with lowest coupling in-degrees
            available_sources = list(range(N1))
            connections_made = 0
            
            # Sort by current in-degree (how many connections from subgraph 2)
            current_in_degrees_1 = np.sum(coupling_matrix.T, axis=1)  # Sum along columns of transpose
            available_sources.sort(key=lambda x: current_in_degrees_1[x])
            
            for i in available_sources:
                if connections_made >= needed_connections:
                    break
                    
                # Add connection if not already present
                if coupling_matrix[i, j] == 0:
                    coupling_matrix[i, j] = 1
                    connections_made += 1
                    
                    # Limit to prevent excessive connections to any single node
                    current_in_degrees_1[i] += 1
                    if current_in_degrees_1[i] >= (target_total_edges // N1) + 2:
                        continue
    
    return coupling_matrix


def create_balanced_coupling_fully_symmetric(N1, N2, l):
    """
    Alternative implementation: Creates a coupling matrix that when made symmetric 
    gives approximately equal coupling degrees to both subgraphs.
    
    Args:
        N1 (int): Size of first subgraph
        N2 (int): Size of second subgraph  
        l (int): Target coupling connections per node
        
    Returns:
        numpy.ndarray: Coupling matrix (N1 x N2) designed for symmetric usage
    """
    if N1 == N2:
        # Equal sized subgraphs - use symmetric approach directly
        return create_balanced_coupling_equal_sizes(N1, N2, l)
    
    coupling_matrix = np.zeros((N1, N2), dtype=int)
    
    # Calculate total connections needed for balance
    total_possible_connections = N1 * N2
    target_connections = min(l * min(N1, N2), total_possible_connections)
    
    # Use random assignment but with balancing constraints
    np.random.seed(42)  # For reproducibility
    
    # Create list of all possible connections
    all_connections = [(i, j) for i in range(N1) for j in range(N2)]
    np.random.shuffle(all_connections)
    
    # Track degrees
    out_degrees_1 = np.zeros(N1, dtype=int)
    in_degrees_2 = np.zeros(N2, dtype=int)
    
    # Maximum connections per node to maintain balance
    max_out_1 = (target_connections // N1) + 2
    max_in_2 = (target_connections // N2) + 2
    
    connections_added = 0
    for i, j in all_connections:
        if connections_added >= target_connections:
            break
            
        # Check if adding this connection maintains balance
        if (out_degrees_1[i] < max_out_1 and 
            in_degrees_2[j] < max_in_2 and
            out_degrees_1[i] < l + 1):  # Don't exceed target too much
            
            coupling_matrix[i, j] = 1
            out_degrees_1[i] += 1
            in_degrees_2[j] += 1
            connections_added += 1
    
    return coupling_matrix


def create_balanced_coupling_equal_sizes(N1, N2, l):
    """
    Optimized version for equal-sized subgraphs (N1 = N2).
    Creates exactly symmetric coupling.
    
    Args:
        N1 (int): Size of first subgraph
        N2 (int): Size of second subgraph (should equal N1)  
        l (int): Number of coupling connections per node
        
    Returns:
        numpy.ndarray: Coupling matrix (N1 x N2)
    """
    if N1 != N2:
        raise ValueError("This function requires N1 = N2")
    
    N = N1
    coupling_matrix = np.zeros((N, N), dtype=int)
    
    # For each node, connect to l other nodes in round-robin fashion
    for i in range(N):
        for k in range(l):
            j = (i + k + 1) % N  # Avoid self-connections, distribute evenly
            coupling_matrix[i, j] = 1
    
    return coupling_matrix


# Updated main coupling function that automatically chooses the best approach
def create_balanced_coupling(N1, N2, l):
    """
    Updated balanced coupling that ensures symmetric coupling degrees.
    Automatically chooses the best approach based on subgraph sizes.
    
    Args:
        N1 (int): Size of first subgraph
        N2 (int): Size of second subgraph  
        l (int): Target number of coupling connections per node
        
    Returns:
        numpy.ndarray: Coupling matrix (N1 x N2)
    """
    if N1 == N2:
        # Use optimized equal-size version
        return create_balanced_coupling_equal_sizes(N1, N2, l)
    else:
        # Use balanced approach for different sizes
        return create_balanced_coupling_symmetric(N1, N2, l)


def visualize_ql_bit(ql_matrix, graph_info, show_plot=True, save_path=None):
    """
    Enhanced visualization of quantum-like bit with flexible coupling support.
    
    Args:
        ql_matrix (numpy.ndarray): (N1+N2) x (N1+N2) adjacency matrix of the quantum-like bit
        graph_info (dict): Information about the graph structure
        show_plot (bool): Whether to display the plot
        save_path (str): Path to save the plot (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    N1 = graph_info['subgraph1_size']
    N2 = graph_info['subgraph2_size']
    k1 = graph_info['subgraph1_degree']
    k2 = graph_info['subgraph2_degree']
    coupling_degree = graph_info['coupling_degree']
    coupling_type = graph_info['coupling_type']
    total_coupling_edges = graph_info['coupling_edges']
    
    # Create NetworkX graph from adjacency matrix
    G = nx.from_numpy_array(ql_matrix)
    
    # Scaling parameters based on graph sizes
    max_size = max(N1, N2)
    if max_size <= 6:
        fig_width, fig_height = 12, 6
        node_size = 600
        font_size = 12
        internal_edge_width = 6
        coupling_edge_width = 2
        separation_distance = 2.5
    elif max_size <= 15:
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
    subgraph1_nodes = list(range(N1))
    subgraph2_nodes = list(range(N1, N1 + N2))
    
    # Create positions - place subgraphs side by side
    pos = {}
    
    # Position subgraph 1 on the left using spring layout
    subgraph1 = G.subgraph(subgraph1_nodes)
    pos1 = nx.spring_layout(subgraph1, seed=42)
    for node, (x, y) in pos1.items():
        pos[node] = (x - separation_distance, y)
    
    # Position subgraph 2 on the right using spring layout
    subgraph2 = G.subgraph(subgraph2_nodes)
    pos2 = nx.spring_layout(subgraph2, seed=43)
    for node, (x, y) in pos2.items():
        pos[node] = (x + separation_distance, y)
    
    # Define edge lists by type
    internal_edges_1 = [(u, v) for u, v in G.edges() if u in subgraph1_nodes and v in subgraph1_nodes]
    internal_edges_2 = [(u, v) for u, v in G.edges() if u in subgraph2_nodes and v in subgraph2_nodes]
    coupling_edges = [(u, v) for u, v in G.edges() if 
                     (u in subgraph1_nodes and v in subgraph2_nodes) or 
                     (u in subgraph2_nodes and v in subgraph1_nodes)]
    
    # Draw edges with different styles and colors
    # 1. Draw all edges first (thin, light gray background for context)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, 
                          edge_color='lightgray', ax=ax)
    
    # 2. Draw internal edges for subgraph 1 (thick, red)
    if internal_edges_1:
        nx.draw_networkx_edges(G, pos, edgelist=internal_edges_1,
                              width=internal_edge_width, alpha=0.6, 
                              edge_color='tab:red', ax=ax)
    
    # 3. Draw internal edges for subgraph 2 (thick, blue)
    if internal_edges_2:
        nx.draw_networkx_edges(G, pos, edgelist=internal_edges_2,
                              width=internal_edge_width, alpha=0.6, 
                              edge_color='tab:blue', ax=ax)
    
    # 4. Draw coupling edges (medium width, black)
    if coupling_edges:
        nx.draw_networkx_edges(G, pos, edgelist=coupling_edges,
                              width=coupling_edge_width, alpha=0.8, 
                              edge_color='black', ax=ax)
    
    # Draw nodes with consistent styling
    node_options = {"edgecolors": "tab:gray", "alpha": 0.9}
    
    # Draw subgraph 1 nodes (red) with size proportional to N1
    nx.draw_networkx_nodes(G, pos, nodelist=subgraph1_nodes, 
                          node_color="tab:red", node_size=node_size, **node_options, ax=ax)
    
    # Draw subgraph 2 nodes (blue) with size proportional to N2
    nx.draw_networkx_nodes(G, pos, nodelist=subgraph2_nodes, 
                          node_color="tab:blue", node_size=node_size, **node_options, ax=ax)
    
    # Draw labels with mathematical notation
    labels = {}
    for i in range(N1):
        labels[i] = f'$a_{{{i}}}$'  # Mathematical notation for subgraph 1
    for i in range(N1, N1 + N2):
        labels[i] = f'$b_{{{i-N1}}}$'  # Mathematical notation for subgraph 2
    
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=font_size, font_color="white", 
                           font_weight='bold', ax=ax)
    
    # Create adaptive title based on coupling type
    if coupling_type == "balanced":
        title = f"Quantum-Like Bit: {k1}-Regular ({N1} nodes) & {k2}-Regular ({N2} nodes) with {coupling_degree} Coupling Edges per Node"
    elif coupling_type == "complete":
        title = f"Quantum-Like Bit: {k1}-Regular ({N1} nodes) & {k2}-Regular ({N2} nodes) with Complete Coupling"
    elif coupling_type == "custom":
        title = f"Quantum-Like Bit: {k1}-Regular ({N1} nodes) & {k2}-Regular ({N2} nodes) with Custom Coupling ({total_coupling_edges} edges)"
    elif coupling_type == "matrix":
        title = f"Quantum-Like Bit: {k1}-Regular ({N1} nodes) & {k2}-Regular ({N2} nodes) with Matrix Coupling ({total_coupling_edges} edges)"
    else:
        title = f"Quantum-Like Bit: {k1}-Regular ({N1} nodes) & {k2}-Regular ({N2} nodes) with {total_coupling_edges} Coupling Edges"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Clean legend
    from matplotlib.patches import Patch
    import matplotlib.lines as mlines
    
    legend_elements = [
        Patch(facecolor='tab:red', edgecolor='tab:gray', label=f'Subgraph 1 ({N1} a-nodes, {k1}-regular)'),
        Patch(facecolor='tab:blue', edgecolor='tab:gray', label=f'Subgraph 2 ({N2} b-nodes, {k2}-regular)'),
        mlines.Line2D([], [], color='tab:red', linewidth=internal_edge_width, 
                     label='Subgraph 1 internal edges'),
        mlines.Line2D([], [], color='tab:blue', linewidth=internal_edge_width, 
                     label='Subgraph 2 internal edges'),
        mlines.Line2D([], [], color='black', linewidth=coupling_edge_width, 
                     label=f'Coupling edges ({total_coupling_edges} total)')
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
