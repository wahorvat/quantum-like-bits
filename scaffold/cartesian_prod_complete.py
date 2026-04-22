import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ql_bit_generator import generate_quantum_like_bit

def generate_complete_graph(m):
    """
    Generate adjacency matrix for a complete graph K_m.
    
    Args:
        m (int): Number of nodes in the complete graph
        
    Returns:
        numpy.ndarray: m x m adjacency matrix of the complete graph
    """
    if m <= 0:
        raise ValueError("Complete graph size m must be positive")
    
    # Complete graph: all nodes connected except to themselves
    return np.ones((m, m), dtype=int) - np.eye(m, dtype=int)

def cartesian_product_ql_complete_variable_l(ql_bit_base, m, l_values, N, k):
    """
    Compute a modified Cartesian product where different l values control 
    connections between QL-bit copies on complete graph nodes.
    
    Args:
        ql_bit_base (numpy.ndarray): Base QL-bit adjacency matrix 
        m (int): Number of complete graph nodes (QL-bit copies)
        l_values (list or numpy.ndarray): l values for pairs of nodes
                 - If list of length m: l_values[i] applies to node i
                 - If m×m matrix: l_values[i,j] applies to pair (i,j)  
        N (int): QL-bit parameter N
        k (int): QL-bit parameter k
        
    Returns:
        numpy.ndarray: Adjacency matrix of the modified Cartesian product
    """
    if m <= 0:
        raise ValueError("Complete graph size m must be positive")
    
    # FIXED: Handle both matrix and tuple inputs
    if isinstance(ql_bit_base, tuple):
        ql_bit_matrix = ql_bit_base[0]  # Extract matrix from tuple
    else:
        ql_bit_matrix = ql_bit_base  # Already a matrix
    
    n = ql_bit_matrix.shape[0]  # FIXED: Use shape[0] instead of len()
    total_size = m * n
    
    # Initialize the product matrix
    product_matrix = np.zeros((total_size, total_size), dtype=int)
    
    # Convert l_values to appropriate format
    if isinstance(l_values, list):
        if len(l_values) == m:
            # l_values per node - create symmetric matrix
            l_matrix = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    if i != j:
                        l_matrix[i, j] = max(l_values[i], l_values[j])  # Use max or mean
        elif len(l_values) == m * (m - 1) // 2:
            # Upper triangular list for pairs
            l_matrix = np.zeros((m, m))
            idx = 0
            for i in range(m):
                for j in range(i + 1, m):
                    l_matrix[i, j] = l_matrix[j, i] = l_values[idx]
                    idx += 1
        else:
            raise ValueError(f"l_values list must have length {m} or {m*(m-1)//2}")
    else:
        # Assume it's an m×m matrix
        l_matrix = np.array(l_values)
        if l_matrix.shape != (m, m):
            raise ValueError(f"l_values matrix must be {m}×{m}")
    
    # Step 1: Add intra-copy edges (QL-bit structure within each copy)
    for i in range(m):
        start_idx = i * n
        end_idx = (i + 1) * n
        product_matrix[start_idx:end_idx, start_idx:end_idx] = ql_bit_matrix  # FIXED: Use matrix directly
    
    # Step 2: Add inter-copy edges based on l_values
    for i in range(m):
        for j in range(i + 1, m):  # Only upper triangle, since undirected
            l_ij = l_matrix[i, j]
            
            if l_ij > 0:
                # Generate connections between copy i and copy j based on l_ij
                start_i, end_i = i * n, (i + 1) * n
                start_j, end_j = j * n, (j + 1) * n
                
                # Create connections based on l_ij parameter
                # Option 1: Local connections within distance l_ij
                for node_a in range(n):
                    for node_b in range(max(0, node_a - int(l_ij)), 
                                       min(n, node_a + int(l_ij) + 1)):
                        if node_a != node_b:
                            # Connect node_a in copy i to node_b in copy j
                            product_matrix[start_i + node_a, start_j + node_b] = 1
                            product_matrix[start_j + node_b, start_i + node_a] = 1
    
    return product_matrix.astype(int)

def cartesian_product_ql_complete_l_matrix(ql_bit_generators, l_matrix):
    """
    Alternative implementation: Generate different QL-bits for each pair.
    
    Args:
        ql_bit_generators (list): List of functions that generate QL-bits
        l_matrix (numpy.ndarray): m×m matrix of l values for pairs
        
    Returns:
        numpy.ndarray: Adjacency matrix of the variable-l Cartesian product
    """
    m = l_matrix.shape[0]
    
    # Assume all QL-bits have the same size (get from first one)
    sample_result = ql_bit_generators[0]()
    # FIXED: Handle tuple return from generator
    if isinstance(sample_result, tuple):
        sample_ql = sample_result[0]
    else:
        sample_ql = sample_result
    
    n = sample_ql.shape[0]
    total_size = m * n
    
    product_matrix = np.zeros((total_size, total_size), dtype=int)
    
    # Add intra-copy edges (each node gets its own QL-bit)
    for i in range(m):
        start_idx = i * n
        end_idx = (i + 1) * n
        ql_result = ql_bit_generators[i]()
        # FIXED: Handle tuple return from generator
        if isinstance(ql_result, tuple):
            ql_bit_i = ql_result[0]
        else:
            ql_bit_i = ql_result
        product_matrix[start_idx:end_idx, start_idx:end_idx] = ql_bit_i
    
    # Add inter-copy edges based on l_matrix
    for i in range(m):
        for j in range(i + 1, m):
            if l_matrix[i, j] > 0:
                start_i, end_i = i * n, (i + 1) * n  
                start_j, end_j = j * n, (j + 1) * n
                
                # Connect all corresponding nodes (complete bipartite between copies)
                # Modified by l_matrix[i,j] parameter
                connection_prob = min(1.0, l_matrix[i, j] / 10.0)  # Scale as needed
                
                for node_a in range(n):
                    for node_b in range(n):
                        if np.random.rand() < connection_prob:
                            product_matrix[start_i + node_a, start_j + node_b] = 1
                            product_matrix[start_j + node_b, start_i + node_a] = 1
                            
    return product_matrix.astype(int)

def cartesian_product_heterogeneous_ql_complete(ql_bit_bases, l_values):
    """
    Optimized Cartesian product where different QL-bits are placed at each node 
    of a complete graph, with variable coupling strengths.
    
    Args:
        ql_bit_bases (list): List of QL-bit adjacency matrices OR tuples (matrix, info), one per complete graph node
        l_values (list or numpy.ndarray): l values controlling connections between QL-bits
    
    Returns:
        numpy.ndarray: Adjacency matrix of the heterogeneous Cartesian product
        list: List of (start_idx, end_idx) tuples indicating where each QL-bit block is located
    """
    m = len(ql_bit_bases)
    if m <= 0:
        raise ValueError("Must provide at least one QL-bit")
    
    # FIXED: Extract matrices from tuples if necessary
    ql_bit_matrices = []
    for ql_bit in ql_bit_bases:
        if isinstance(ql_bit, tuple):
            ql_bit_matrices.append(ql_bit[0])  # Extract matrix from tuple
        else:
            ql_bit_matrices.append(ql_bit)  # Already a matrix
    
    # Get sizes of each QL-bit
    ql_sizes = np.array([ql_bit.shape[0] for ql_bit in ql_bit_matrices])
    total_size = int(np.sum(ql_sizes))
    
    # Validate that all QL-bits are square matrices
    for i, ql_bit in enumerate(ql_bit_matrices):
        if ql_bit.shape[0] != ql_bit.shape[1]:
            raise ValueError(f"QL-bit {i} must be a square matrix")
    
    # Check for homogeneous case (all QL-bits identical) for optimization
    all_identical = all(np.array_equal(ql_bit_matrices[0], ql_bit) for ql_bit in ql_bit_matrices[1:])
    
    if all_identical:
        # Use optimized homogeneous path
        return _optimized_homogeneous_path(ql_bit_matrices[0], m, l_values)
    
    # Calculate block boundaries using cumsum for efficiency
    block_starts = np.concatenate([[0], np.cumsum(ql_sizes)[:-1]])
    block_ends = np.cumsum(ql_sizes)
    block_boundaries = list(zip(block_starts, block_ends))
    
    # Convert l_values to matrix format efficiently
    l_matrix = _convert_l_values_fast(l_values, m)
    
    # Pre-allocate the product matrix
    product_matrix = np.zeros((total_size, total_size), dtype=np.int8)
    
    # Step 1: Add intra-QL-bit edges using vectorized assignment
    for i, (start_i, end_i) in enumerate(block_boundaries):
        product_matrix[start_i:end_i, start_i:end_i] = ql_bit_matrices[i]
    
    # Step 2: Pre-compute and cache coupling matrices to avoid redundant calculations
    unique_size_pairs = {}
    unique_l_values = set()
    
    # Find unique (size1, size2, l_value) combinations
    for i in range(m):
        for j in range(i + 1, m):
            l_ij = l_matrix[i, j]
            if l_ij > 0:
                size_i, size_j = ql_sizes[i], ql_sizes[j]
                key = (min(size_i, size_j), max(size_i, size_j), int(l_ij))
                unique_size_pairs[key] = None
                unique_l_values.add(int(l_ij))
    
    # Pre-compute all unique coupling matrices
    for (size1, size2, l_val) in unique_size_pairs:
        unique_size_pairs[(size1, size2, l_val)] = create_heterogeneous_coupling_fast(
            size1, size2, l_val)
    
    # Step 3: Apply coupling matrices efficiently
    for i in range(m):
        start_i, end_i = block_boundaries[i]
        for j in range(i + 1, m):
            l_ij = l_matrix[i, j]
            if l_ij > 0:
                start_j, end_j = block_boundaries[j]
                size_i, size_j = ql_sizes[i], ql_sizes[j]
                
                # Retrieve pre-computed coupling matrix
                key = (min(size_i, size_j), max(size_i, size_j), int(l_ij))
                coupling_matrix = unique_size_pairs[key]
                
                # Handle size ordering
                if size_i <= size_j:
                    product_matrix[start_i:end_i, start_j:end_j] = coupling_matrix
                    product_matrix[start_j:end_j, start_i:end_i] = coupling_matrix.T
                else:
                    product_matrix[start_i:end_i, start_j:end_j] = coupling_matrix.T
                    product_matrix[start_j:end_j, start_i:end_i] = coupling_matrix
    
    return product_matrix.astype(int), block_boundaries

def _optimized_homogeneous_path(ql_bit_base, m, l_values):
    """Optimized path when all QL-bits are identical."""
    n = ql_bit_base.shape[0]  # FIXED: Use shape[0] instead of len()
    total_size = m * n
    
    # Convert l_values to matrix
    l_matrix = _convert_l_values_fast(l_values, m)
    
    # Use Kronecker product approach for identical blocks
    I_m = np.eye(m, dtype=np.int8)
    I_n = np.eye(n, dtype=np.int8)
    
    # Intra-QL-bit connections: I_m ⊗ QL_bit
    product_matrix = np.kron(I_m, ql_bit_base).astype(np.int8)
    
    # Pre-compute coupling matrix once
    unique_l_vals = np.unique(l_matrix[l_matrix > 0])
    coupling_cache = {}
    for l_val in unique_l_vals:
        coupling_cache[l_val] = create_heterogeneous_coupling_fast(n, n, int(l_val))
    
    # Add inter-QL-bit connections efficiently
    for i in range(m):
        start_i = i * n
        end_i = (i + 1) * n
        for j in range(i + 1, m):
            l_ij = l_matrix[i, j]
            if l_ij > 0:
                start_j = j * n
                end_j = (j + 1) * n
                
                coupling_matrix = coupling_cache[l_ij]
                product_matrix[start_i:end_i, start_j:end_j] = coupling_matrix
                product_matrix[start_j:end_j, start_i:end_i] = coupling_matrix.T
    
    # Create block boundaries
    block_boundaries = [(i * n, (i + 1) * n) for i in range(m)]
    
    return product_matrix.astype(int), block_boundaries

def _convert_l_values_fast(l_values, m):
    """Fast conversion of l_values to matrix format."""
    if isinstance(l_values, np.ndarray):
        if l_values.shape == (m, m):
            return l_values
        else:
            raise ValueError(f"l_values array must be {m}×{m}")
    
    if isinstance(l_values, list):
        if len(l_values) > 0 and isinstance(l_values[0], (list, np.ndarray)):
            l_matrix = np.array(l_values)
            if l_matrix.shape != (m, m):
                raise ValueError(f"l_values matrix must be {m}×{m}")
            return l_matrix
        elif len(l_values) == m:
            # Vectorized creation of symmetric matrix
            l_array = np.array(l_values)
            l_matrix = np.maximum(l_array[:, None], l_array[None, :])
            np.fill_diagonal(l_matrix, 0)
            return l_matrix
        elif len(l_values) == m * (m - 1) // 2:
            l_matrix = np.zeros((m, m))
            idx = 0
            for i in range(m):
                for j in range(i + 1, m):
                    l_matrix[i, j] = l_matrix[j, i] = l_values[idx]
                    idx += 1
            return l_matrix
    
    raise ValueError(f"Invalid l_values format")

def create_heterogeneous_coupling_fast(size_i, size_j, l_coupling, seed=42):
    """
    Fast creation of balanced coupling matrix between QL-bits of different sizes.
    
    Args:
        size_i (int): Size of first QL-bit
        size_j (int): Size of second QL-bit  
        l_coupling (int): Number of connections per node (approximate)
        seed (int): Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Coupling matrix (size_i × size_j)
    """
    if l_coupling <= 0:
        return np.zeros((size_i, size_j), dtype=np.int8)
    
    # Set random seed for reproducibility
    np.random.seed(seed + size_i + size_j + l_coupling)
    
    # Pre-allocate coupling matrix
    coupling_matrix = np.zeros((size_i, size_j), dtype=np.int8)
    
    if l_coupling >= size_j:
        # If l_coupling >= size_j, connect each node in i to all nodes in j
        coupling_matrix.fill(1)
        return coupling_matrix
    
    # Use deterministic pattern for better performance and reproducibility
    # Create a more efficient connection pattern
    for i in range(size_i):
        # Deterministic selection of l_coupling targets for node i
        # Use modular arithmetic to ensure balanced distribution
        targets = []
        for k in range(l_coupling):
            target = (i * l_coupling + k) % size_j
            if target not in targets:  # Avoid duplicates
                targets.append(target)
        
        # Fill remaining if we had duplicates
        offset = 0
        while len(targets) < l_coupling and len(targets) < size_j:
            candidate = (i * l_coupling + l_coupling + offset) % size_j
            if candidate not in targets:
                targets.append(candidate)
            offset += 1
        
        # Set connections
        for target in targets:
            coupling_matrix[i, target] = 1
    
    return coupling_matrix

def analyze_heterogeneous_spectrum(ql_bit_bases, l_values, show_details=True):
    """
    Analyze the spectrum of a heterogeneous Cartesian product.
    
    Args:
        ql_bit_bases (list): List of QL-bit adjacency matrices or tuples
        l_values: Coupling parameters
        show_details (bool): Whether to show detailed eigenvalue analysis
    
    Returns:
        tuple: (eigenvalues, eigenvectors, block_boundaries)
    """
    # FIXED: Extract matrices if needed
    ql_bit_matrices = []
    for ql_bit in ql_bit_bases:
        if isinstance(ql_bit, tuple):
            ql_bit_matrices.append(ql_bit[0])
        else:
            ql_bit_matrices.append(ql_bit)
    
    # Compute the heterogeneous Cartesian product
    product_matrix, block_boundaries = cartesian_product_heterogeneous_ql_complete(
        ql_bit_matrices, l_values)
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(product_matrix.astype(float))
    
    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    if show_details:
        print("HETEROGENEOUS CARTESIAN PRODUCT ANALYSIS")
        print("=" * 50)
        print(f"Number of different QL-bits: {len(ql_bit_matrices)}")
        print(f"QL-bit sizes: {[ql.shape[0] for ql in ql_bit_matrices]}")
        print(f"Total system size: {product_matrix.shape[0]}")
        print(f"Block boundaries: {block_boundaries}")
        print()
        
        print("Individual QL-bit spectra:")
        for i, ql_bit in enumerate(ql_bit_matrices):
            ql_eigenvals = np.linalg.eigvals(ql_bit.astype(float))
            ql_eigenvals = np.sort(ql_eigenvals)[::-1]
            print(f"  QL-bit {i} (size {ql_bit.shape[0]}): {ql_eigenvals[:5]}...")
        print()
        
        print("Combined system spectrum:")
        print(f"  Largest eigenvalue: {eigenvals[0]:.6f}")
        print(f"  Smallest eigenvalue: {eigenvals[-1]:.6f}")
        print(f"  Spectral radius: {np.max(np.abs(eigenvals)):.6f}")
        print(f"  Number of unique eigenvalues: {len(np.unique(np.round(eigenvals, 8)))}")
        print(f"  Top 20 eigenvalues: {eigenvals[:20]}")
        print()
        
        # Analyze eigenvalue degeneracies
        unique_eigenvals, counts = np.unique(np.round(eigenvals, 8), return_counts=True)
        degenerate_eigenvals = [(eig, count) for eig, count in zip(unique_eigenvals, counts) if count > 1]
        
        if degenerate_eigenvals:
            print(f"Degenerate eigenvalues ({len(degenerate_eigenvals)} found):")
            for eig, mult in degenerate_eigenvals[:10]:  # Show first 10
                print(f"  λ = {eig:.6f}, multiplicity = {mult}")
        else:
            print("No degenerate eigenvalues found.")
    
    return eigenvals, eigenvecs, block_boundaries

def visualize_heterogeneous_structure(ql_bit_bases, l_values, eigenvals=None):
    """
    Visualize the structure of a heterogeneous Cartesian product.
    """
    # FIXED: Extract matrices if needed
    ql_bit_matrices = []
    for ql_bit in ql_bit_bases:
        if isinstance(ql_bit, tuple):
            ql_bit_matrices.append(ql_bit[0])
        else:
            ql_bit_matrices.append(ql_bit)
    
    product_matrix, block_boundaries = cartesian_product_heterogeneous_ql_complete(
        ql_bit_matrices, l_values)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Adjacency matrix structure
    im1 = axes[0].imshow(product_matrix, cmap='RdBu_r', aspect='equal')
    axes[0].set_title('Heterogeneous Cartesian Product\nAdjacency Matrix')
    axes[0].set_xlabel('Node index')
    axes[0].set_ylabel('Node index')
    
    # Add block boundary lines
    for start, end in block_boundaries[1:]:
        axes[0].axhline(y=start-0.5, color='yellow', linewidth=2, alpha=0.7)
        axes[0].axvline(x=start-0.5, color='yellow', linewidth=2, alpha=0.7)
    
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Plot 2: Block structure diagram
    axes[1].set_xlim(-1, len(ql_bit_matrices))
    axes[1].set_ylim(-1, max([ql.shape[0] for ql in ql_bit_matrices]) + 1)
    
    for i, (ql_bit, (start, end)) in enumerate(zip(ql_bit_matrices, block_boundaries)):
        size = ql_bit.shape[0]
        # Draw QL-bit block
        rect = plt.Rectangle((i-0.4, 0), 0.8, size, 
                           fill=False, edgecolor='blue', linewidth=2)
        axes[1].add_patch(rect)
        axes[1].text(i, size/2, f'QL-bit {i}\n(size {size})', 
                    ha='center', va='center', fontweight='bold')
        
        # Draw connections to other QL-bits
        for j in range(i+1, len(ql_bit_matrices)):
            axes[1].plot([i+0.4, j-0.4], [size/2, ql_bit_matrices[j].shape[0]/2], 
                        'r--', alpha=0.6, linewidth=1)
    
    axes[1].set_title('QL-bit Block Structure')
    axes[1].set_xlabel('QL-bit index')
    axes[1].set_ylabel('QL-bit size')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Eigenvalue spectrum (if provided)
    if eigenvals is not None:
        axes[2].hist(eigenvals, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[2].set_xlabel('Eigenvalue')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Eigenvalue Distribution')
        axes[2].grid(True, alpha=0.3)
        
        # Mark largest and smallest eigenvalues
        axes[2].axvline(eigenvals[0], color='red', linestyle='--', 
                       label=f'Max: {eigenvals[0]:.3f}')
        axes[2].axvline(eigenvals[-1], color='blue', linestyle='--', 
                       label=f'Min: {eigenvals[-1]:.3f}')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'Eigenvalue spectrum\nnot computed', 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[2].set_title('Eigenvalue Distribution')
    
    plt.tight_layout()
    plt.show()


def visualize_cartesian_product(product_matrix, m, N):
    """
    Visualize the Cartesian product K_m □ QL-bit.
    Places QL-bits at vertices of a regular polygon representing the complete graph.
    
    Args:
        product_matrix (numpy.ndarray): Adjacency matrix of the Cartesian product
        m (int): Size of the complete graph
        N (int): Size of each QL-bit subgraph
    """
    # Create NetworkX graph
    G = nx.from_numpy_array(product_matrix)
    
    # Node organization: For K_m □ QL-bit, nodes are indexed as (k, q)
    # where k ∈ {0, 1, ..., m-1} and q ∈ {0, 1, ..., 2N-1}
    # Node index = k * (2N) + q
    
    total_nodes = m * 2 * N
    pos = {}
    
    # Step 1: Create regular polygon positions for the complete graph vertices
    main_radius = max(3, m * 0.8)  # Scale radius based on m
    complete_graph_positions = []
    
    if m == 1:
        complete_graph_positions = [(0, 0)]
    else:
        angles = np.linspace(0, 2*np.pi, m, endpoint=False)
        for angle in angles:
            complete_graph_positions.append((main_radius * np.cos(angle), 
                                           main_radius * np.sin(angle)))
    
    # Step 2: At each complete graph vertex, place a QL-bit
    ql_bit_radius = min(1.0, main_radius / (2 * m))  # Scale QL-bit size
    
    for k in range(m):  # For each complete graph node
        center_x, center_y = complete_graph_positions[k]
        
        # Subgraph 1 nodes (a-nodes) - small circle above center
        subgraph1_nodes = [k * (2*N) + i for i in range(N)]
        if N > 1:
            angles1 = np.linspace(0, 2*np.pi, N, endpoint=False)
            for i, node in enumerate(subgraph1_nodes):
                pos[node] = (center_x + ql_bit_radius * np.cos(angles1[i]), 
                           center_y + ql_bit_radius + ql_bit_radius * np.sin(angles1[i]))
        else:
            pos[subgraph1_nodes[0]] = (center_x, center_y + ql_bit_radius)
        
        # Subgraph 2 nodes (b-nodes) - small circle below center  
        subgraph2_nodes = [k * (2*N) + N + i for i in range(N)]
        if N > 1:
            angles2 = np.linspace(0, 2*np.pi, N, endpoint=False)
            for i, node in enumerate(subgraph2_nodes):
                pos[node] = (center_x + ql_bit_radius * np.cos(angles2[i]), 
                           center_y - ql_bit_radius + ql_bit_radius * np.sin(angles2[i]))
        else:
            pos[subgraph2_nodes[0]] = (center_x, center_y - ql_bit_radius)
    
    # Create figure with appropriate size
    fig_size = max(8, main_radius * 0.8)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Define node colors and labels
    node_colors = []
    node_labels = {}
    
    for node in range(total_nodes):
        k = node // (2*N)  # Which complete graph node
        q = node % (2*N)   # Which QL-bit node
        
        if q < N:  # Subgraph 1 (a-nodes)
            node_colors.append('lightcoral')
            node_labels[node] = f'a{q}' if total_nodes <= 30 else ''
        else:  # Subgraph 2 (b-nodes)
            node_colors.append('lightblue') 
            node_labels[node] = f'b{q-N}' if total_nodes <= 30 else ''
    
    # Step 3: Draw edges with different styles based on type
    # Separate edges by type for better visualization
    ql_internal_edges_1 = []  # Within subgraph 1 of same QL-bit
    ql_internal_edges_2 = []  # Within subgraph 2 of same QL-bit  
    ql_coupling_edges = []    # Between subgraphs of same QL-bit
    complete_graph_edges = [] # Between corresponding nodes of different QL-bits
    
    for u, v in G.edges():
        k1, q1 = u // (2*N), u % (2*N)
        k2, q2 = v // (2*N), v % (2*N)
        
        if k1 == k2:  # Same QL-bit copy
            if q1 < N and q2 < N:  # Both in subgraph 1
                ql_internal_edges_1.append((u, v))
            elif q1 >= N and q2 >= N:  # Both in subgraph 2
                ql_internal_edges_2.append((u, v))
            else:  # Coupling between subgraphs
                ql_coupling_edges.append((u, v))
        else:  # Different QL-bit copies (complete graph connections)
            complete_graph_edges.append((u, v))
    
    # Draw edges with different styles
    # 1. Complete graph edges (thin, light gray, in background)
    if complete_graph_edges:
        nx.draw_networkx_edges(G, pos, edgelist=complete_graph_edges,
                              edge_color='lightgray', width=0.5, alpha=0.4)
    
    # 2. QL-bit internal edges (thicker, colored)
    if ql_internal_edges_1:
        nx.draw_networkx_edges(G, pos, edgelist=ql_internal_edges_1,
                              edge_color='red', width=2, alpha=0.7)
    if ql_internal_edges_2:
        nx.draw_networkx_edges(G, pos, edgelist=ql_internal_edges_2,
                              edge_color='blue', width=2, alpha=0.7)
    
    # 3. QL-bit coupling edges (medium, black, dashed)
    if ql_coupling_edges:
        nx.draw_networkx_edges(G, pos, edgelist=ql_coupling_edges,
                              edge_color='black', width=1.5, alpha=0.8, style='dashed')
    
    # Draw nodes
    node_size = max(100, 500 / max(1, total_nodes / 10))  # Scale node size
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, 
                          edgecolors='gray', linewidths=1, alpha=0.9)
    
    # Add labels (only if not too crowded)
    if total_nodes <= 30:
        font_size = max(6, 12 - total_nodes // 5)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=font_size, 
                               font_weight='bold', font_color='white')
    
    # Draw complete graph vertex labels at centers
    if m <= 10:
        center_labels = {}
        for k in range(m):
            center_x, center_y = complete_graph_positions[k]
            # Add a text label at the center of each QL-bit
            plt.text(center_x, center_y, f'K{k}', fontsize=12, fontweight='bold',
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Title and legend
    plt.title(f'Cartesian Product: K_{m} □ QL-bit\n'
              f'QL-bits positioned at vertices of K_{m} (N={N} per subgraph)\n'
              f'Total nodes: {total_nodes}, Edges: {np.sum(product_matrix) // 2}', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Create legend
    from matplotlib.patches import Patch
    import matplotlib.lines as mlines
    
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', alpha=0.7, label=f'K_{m} vertex centers'),
        Patch(facecolor='lightcoral', edgecolor='gray', label='Subgraph 1 (a-nodes)'),
        Patch(facecolor='lightblue', edgecolor='gray', label='Subgraph 2 (b-nodes)'),
        mlines.Line2D([], [], color='red', linewidth=2, label='Internal edges (subgraph 1)'),
        mlines.Line2D([], [], color='blue', linewidth=2, label='Internal edges (subgraph 2)'),
        mlines.Line2D([], [], color='black', linewidth=1.5, linestyle='--', label='Coupling edges'),
        mlines.Line2D([], [], color='lightgray', linewidth=0.5, label=f'K_{m} connections')
    ]
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.axis('off')
    plt.axis('equal')  # Maintain aspect ratio
    plt.tight_layout()
    plt.show()

def visualize_eigenvalues(ql_bit_matrix, cartesian_product_matrix, m, N):
    """
    Visualize the eigenvalue distributions of the QL-bit, complete graph, and their Cartesian product.
    Uses log scale with single eigenvalues visible at 1/4 height.
    
    Args:
        ql_bit_matrix (numpy.ndarray): Adjacency matrix of the quantum-like bit
        cartesian_product_matrix (numpy.ndarray): Adjacency matrix of K_m □ QL-bit
        m (int): Size of the complete graph
        N (int): Size of each QL-bit subgraph
    """
    # FIXED: Handle tuple input for ql_bit_matrix
    if isinstance(ql_bit_matrix, tuple):
        ql_bit_matrix = ql_bit_matrix[0]
    
    # Generate complete graph adjacency matrix
    complete_graph_matrix = np.ones((m, m), dtype=int) - np.eye(m, dtype=int)
    
    # Compute eigenvalues and eigenvectors
    ql_eigenvalues, ql_eigenvectors = np.linalg.eigh(ql_bit_matrix.astype(float))
    complete_eigenvalues, complete_eigenvectors = np.linalg.eigh(complete_graph_matrix.astype(float))
    product_eigenvalues, product_eigenvectors = np.linalg.eigh(cartesian_product_matrix.astype(float))
    
    # Sort eigenvalues and eigenvectors for better visualization (descending order)
    ql_idx = np.argsort(ql_eigenvalues)[::-1]
    ql_eigenvalues = ql_eigenvalues[ql_idx]
    ql_eigenvectors = ql_eigenvectors[:, ql_idx]
    
    complete_idx = np.argsort(complete_eigenvalues)[::-1]
    complete_eigenvalues = complete_eigenvalues[complete_idx]
    complete_eigenvectors = complete_eigenvectors[:, complete_idx]
    
    product_idx = np.argsort(product_eigenvalues)[::-1]
    product_eigenvalues = product_eigenvalues[product_idx]
    product_eigenvectors = product_eigenvectors[:, product_idx]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # Helper function to set up log scale plotting
    def setup_log_histogram(ax, eigenvals, color, edge_color, title):
        # Create histogram
        counts, bin_edges, patches = ax.hist(eigenvals, bins=100, color=color, 
                                           alpha=0.7, edgecolor=edge_color)
        
        # Add small constant to avoid log(0) and make single eigenvalues visible
        offset = 0.1
        counts_with_offset = counts + offset
        
        # Clear the axis and replot with adjusted heights
        ax.clear()
        
        # Replot histogram with offset
        n, bins, patches = ax.hist(eigenvals, bins=100, color=color, 
                                  alpha=0.7, edgecolor=edge_color)
        
        # Adjust the heights to add offset
        for patch, count in zip(patches, n):
            patch.set_height(count + offset)
        
        # Set log scale
        ax.set_yscale('log')
        
        # Set y-axis limits so single eigenvalues (frequency = 1 + offset) appear at 1/4 height
        max_count = np.max(n) + offset
        min_visible = 1.0 + offset  # Single eigenvalue frequency
        
        # Set limits: single eigenvalues at 1/4, max at top
        # log(min_visible) should be at 1/4 of log(max_count * 4)
        y_max = max_count * 4
        y_min = min_visible / 4
        
        ax.set_ylim(y_min, y_max)
        
        # Add scatter points for individual eigenvalues at the 1/4 line
        scatter_y = min_visible  # This will be at 1/4 height visually
        if len(eigenvals) <= 200:
            ax.scatter(eigenvals, np.full_like(eigenvals, scatter_y), 
                      alpha=0.8, s=25, c=edge_color, zorder=5, marker='|')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Eigenvalue')
        ax.set_ylabel('Frequency (log scale)')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        return n, bins
    
    # Plot 1: QL-bit eigenvalues
    n1, bins1 = setup_log_histogram(ax1, ql_eigenvalues, 'red', 'darkred',
                                   f'QL-bit Eigenvalue Distribution\n(2 × {N} subgraphs, {len(ql_eigenvalues)} total)')
    
    # Add statistics box
    ql_stats_text = f'Max: {ql_eigenvalues.max():.3f}\nMin: {ql_eigenvalues.min():.3f}\nSpectral gap: {ql_eigenvalues[0] - ql_eigenvalues[1]:.3f}'
    ax1.text(0.02, 0.98, ql_stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Complete graph eigenvalues
    n2, bins2 = setup_log_histogram(ax2, complete_eigenvalues, 'blue', 'darkblue',
                                   f'Complete Graph K_{m} Eigenvalues\n({len(complete_eigenvalues)} total)')
    
    # Add multiplicity annotations for complete graph
    unique_eigs, counts = np.unique(np.round(complete_eigenvalues, 8), return_counts=True)
    for eig, count in zip(unique_eigs, counts):
        # Position annotation at 1/2 height for visibility
        y_pos = (1.0 + 0.1) * 2  # Twice the single eigenvalue height
        ax2.annotate(f'mult: {count}', xy=(eig, y_pos), xytext=(eig, y_pos * 2),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.7))
    
    # Add statistics box
    complete_stats_text = f'Max: {complete_eigenvalues.max():.3f}\nMin: {complete_eigenvalues.min():.3f}\nSpectral gap: {complete_eigenvalues[0] - complete_eigenvalues[1]:.3f}'
    ax2.text(0.02, 0.98, complete_stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 3: Cartesian product eigenvalues
    n3, bins3 = setup_log_histogram(ax3, product_eigenvalues, 'green', 'darkgreen',
                                   f'K_{m} □ QL-bit Eigenvalue Distribution\n({len(product_eigenvalues)} total)')
    
    # Add statistics box
    product_stats_text = f'Max: {product_eigenvalues.max():.3f}\nMin: {product_eigenvalues.min():.3f}\nSpectral gap: {product_eigenvalues[0] - product_eigenvalues[1]:.3f}'
    ax3.text(0.02, 0.98, product_stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Add horizontal reference lines at key frequency levels
    for ax in [ax1, ax2, ax3]:
        # Line at single eigenvalue level (1/4 height visually)
        ax.axhline(y=1.1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, 1.1, 
               'Single eigenvalue', fontsize=8, va='bottom', alpha=0.7)
    
    # Overall title
    fig.suptitle(f'Eigenvalue Spectra Comparison (Log Scale): QL-bit, K_{m}, and K_{m} □ QL-bit', 
                 fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print eigenvalue and eigenvector information
    print(f"\nEigenvalue Analysis:")
    print(f"QL-bit ({2*N} nodes):")
    print(f"  Largest eigenvalue: {ql_eigenvalues[0]:.6f}")
    print(f"  Smallest eigenvalue: {ql_eigenvalues[-1]:.6f}")
    print(f"  Spectral radius: {np.max(np.abs(ql_eigenvalues)):.6f}")
    print(f"  Unique eigenvalues: {len(np.unique(np.round(ql_eigenvalues, 8)))}")
    print(f"  Top 100 eigenvalues: {ql_eigenvalues[:100]}")
    print(f"  First 5 eigenvectors (columns, top 100 elements):")
    for i in range(min(5, ql_eigenvectors.shape[1])):
        print(f"    Eigenvector {i+1} (λ={ql_eigenvalues[i]:.6f}): {ql_eigenvectors[:100, i]}")
    
    print(f"\nComplete graph K_{m} ({m} nodes):")
    print(f"  Largest eigenvalue: {complete_eigenvalues[0]:.6f}")
    print(f"  Smallest eigenvalue: {complete_eigenvalues[-1]:.6f}")
    print(f"  Spectral radius: {np.max(np.abs(complete_eigenvalues)):.6f}")
    print(f"  Theoretical eigenvalues: {m-1} (mult. 1), -1 (mult. {m-1})")
    print(f"  Top 100 eigenvalues: {complete_eigenvalues[:100]}")
    print(f"  First 5 eigenvectors (columns, top 100 elements):")
    for i in range(min(5, complete_eigenvectors.shape[1])):
        print(f"    Eigenvector {i+1} (λ={complete_eigenvalues[i]:.6f}): {complete_eigenvectors[:100, i]}")
    
    print(f"\nK_{m} □ QL-bit ({len(product_eigenvalues)} nodes):")
    print(f"  Largest eigenvalue: {product_eigenvalues[0]:.6f}")
    print(f"  Smallest eigenvalue: {product_eigenvalues[-1]:.6f}")
    print(f"  Spectral radius: {np.max(np.abs(product_eigenvalues)):.6f}")
    print(f"  Unique eigenvalues: {len(np.unique(np.round(product_eigenvalues, 8)))}")
    print(f"  Top 100 eigenvalues: {product_eigenvalues[:100]}")
    print(f"  First 5 eigenvectors (columns, top 100 elements):")
    for i in range(min(5, product_eigenvectors.shape[1])):
        print(f"    Eigenvector {i+1} (λ={product_eigenvalues[i]:.6f}): {product_eigenvectors[:100, i]}")
    
    # Theoretical relationship for Cartesian product eigenvalues
    print(f"\nTheoretical Note:")
    print(f"For Cartesian product A □ B, eigenvalues are λ_A + λ_B")
    print(f"Expected range: [{ql_eigenvalues.min() + complete_eigenvalues.min():.3f}, "
          f"{ql_eigenvalues.max() + complete_eigenvalues.max():.3f}]")
    
    return ql_eigenvalues, complete_eigenvalues, product_eigenvalues, ql_eigenvectors, complete_eigenvectors, product_eigenvectors