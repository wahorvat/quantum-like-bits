"""
k_regular_graph_generator.py

A module for generating k-regular graphs and their adjacency matrices.
ROBUST VERSION: Handles graphs up to size 1000+ with multiple algorithmic approaches.
"""

import random
import networkx as nx
import numpy as np
from collections import defaultdict


def generate_k_regular_graph(N, k):
    """
    Generate a random k-regular graph on N nodes using robust algorithms.
    
    Args:
        N (int): Number of nodes
        k (int): Degree of each node
    
    Returns:
        networkx.Graph: A k-regular graph
    
    Raises:
        ValueError: If the parameters don't allow for a valid k-regular graph
        RuntimeError: If graph generation fails after maximum attempts
    """
    # Check if k-regular graph is possible
    if k >= N:
        raise ValueError(f"k ({k}) must be less than N ({N})")
    
    if (N * k) % 2 != 0:
        raise ValueError(f"N*k ({N*k}) must be even for a valid graph")
    
    if k < 0:
        raise ValueError("k must be non-negative")
    
    # Special cases
    if k == 0:
        return nx.empty_graph(N)
    
    if k == N - 1:
        return nx.complete_graph(N)
    
    # Choose algorithm based on graph density and size
    density_ratio = k / (N - 1)
    
    if density_ratio > 0.7:  # High density: remove edges from complete graph
        return _generate_by_edge_removal(N, k)
    elif k <= 3 or (k <= 6 and N <= 50):  # Low degree or small graphs: use incremental
        return _generate_by_incremental_improved(N, k)
    else:  # Medium density: use robust configuration model
        return _generate_by_configuration_robust(N, k)


def _generate_by_configuration_robust(N, k):
    """
    Robust configuration model implementation with proper multi-edge and self-loop handling.
    """
    max_attempts = 50
    
    for attempt in range(max_attempts):
        try:
            # Create degree sequence
            degree_sequence = [k] * N
            
            # Generate configuration model graph
            G = nx.configuration_model(degree_sequence, seed=random.randint(0, 1000000))
            
            # Convert to simple graph and handle issues
            G = _clean_configuration_graph(G, k)
            
            if G is not None and _is_k_regular(G, k):
                return G
                
        except Exception:
            continue
    
    # Fallback to edge swapping method
    return _generate_by_edge_swapping(N, k)


def _clean_configuration_graph(G, k):
    """
    Clean configuration model graph by removing self-loops and multi-edges,
    then fixing degree issues through intelligent edge manipulation.
    """
    # Remove self-loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    
    # Remove parallel edges by converting to simple graph
    G = nx.Graph(G)
    
    # Fix degree deficiencies through targeted edge addition
    max_fix_attempts = 200
    attempt = 0
    
    while not _is_k_regular(G, k) and attempt < max_fix_attempts:
        attempt += 1
        degrees = dict(G.degree())
        
        # Find nodes with degree deficiencies
        deficit_nodes = [node for node in G.nodes() if degrees[node] < k]
        excess_nodes = [node for node in G.nodes() if degrees[node] > k]
        
        if deficit_nodes and len(deficit_nodes) >= 2:
            # Add edges between deficit nodes
            random.shuffle(deficit_nodes)
            for i in range(0, len(deficit_nodes) - 1, 2):
                node1, node2 = deficit_nodes[i], deficit_nodes[i + 1]
                if not G.has_edge(node1, node2) and degrees[node1] < k and degrees[node2] < k:
                    G.add_edge(node1, node2)
                    degrees[node1] += 1
                    degrees[node2] += 1
        
        elif excess_nodes and deficit_nodes:
            # Redistribute edges from excess to deficit nodes
            _redistribute_edges(G, excess_nodes, deficit_nodes, k)
        
        elif not deficit_nodes and not excess_nodes:
            break  # Already k-regular
        
        else:
            # Use random edge swapping
            _perform_random_edge_swap(G, k)
    
    return G if _is_k_regular(G, k) else None


def _generate_by_edge_removal(N, k):
    """
    Generate k-regular graph by removing edges from complete graph.
    Optimized for high-density graphs.
    """
    G = nx.complete_graph(N)
    target_edges = (N * k) // 2
    current_edges = G.number_of_edges()
    edges_to_remove = current_edges - target_edges
    
    edges_list = list(G.edges())
    random.shuffle(edges_list)
    
    # Remove edges while maintaining degree constraints
    removed = 0
    for u, v in edges_list:
        if removed >= edges_to_remove:
            break
        if G.degree[u] > k and G.degree[v] > k:
            G.remove_edge(u, v)
            removed += 1
    
    # Fine-tune with edge swapping
    max_swaps = N * 10
    for _ in range(max_swaps):
        if _is_k_regular(G, k):
            break
        _perform_smart_edge_swap(G, k)
    
    if not _is_k_regular(G, k):
        raise RuntimeError("Failed to achieve k-regularity through edge removal")
    
    return G


def _generate_by_incremental_improved(N, k):
    """
    Improved incremental construction for small k values.
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    # Use more systematic approach instead of pure random
    nodes = list(range(N))
    target_degree = k
    
    # Phase 1: Build approximate structure
    for node in nodes:
        current_degree = G.degree[node]
        needed_edges = target_degree - current_degree
        
        # Find best candidates for connections
        candidates = []
        for other_node in nodes:
            if (other_node != node and 
                not G.has_edge(node, other_node) and 
                G.degree[other_node] < target_degree):
                candidates.append(other_node)
        
        # Sort candidates by their current degree (prefer lower-degree nodes)
        candidates.sort(key=lambda x: G.degree[x])
        
        # Add edges to best candidates
        for candidate in candidates[:needed_edges]:
            if G.degree[node] < target_degree and G.degree[candidate] < target_degree:
                G.add_edge(node, candidate)
    
    # Phase 2: Fix any remaining degree issues
    max_fixes = N * 50
    for _ in range(max_fixes):
        if _is_k_regular(G, k):
            break
        _perform_degree_fixing_swap(G, k)
    
    if not _is_k_regular(G, k):
        raise RuntimeError("Failed to achieve k-regularity through incremental construction")
    
    return G


def _generate_by_edge_swapping(N, k):
    """
    Generate k-regular graph using advanced edge swapping starting from a base configuration.
    """
    # Start with a simple regular structure if possible
    if k == 2:
        # Start with a cycle
        G = nx.cycle_graph(N)
    elif k % 2 == 0 and k <= N // 2:
        # Start with union of cycles
        G = _create_base_regular_graph(N, k)
    else:
        # Start with random regular-like structure
        G = _create_approximate_regular_graph(N, k)
    
    # FIX #2: Randomize through edge swapping with reasonable timeout protection
    num_swaps = min(N * k * 2, 10000)  # Cap at 10,000 swaps maximum
    successful_swaps = 0
    attempts = 0
    max_attempts = min(num_swaps * 3, 20000)  # Reduced multiplier and added hard cap
    
    while successful_swaps < num_swaps and attempts < max_attempts:
        attempts += 1
        if _perform_random_edge_swap(G, k):
            successful_swaps += 1
    
    if not _is_k_regular(G, k):
        raise RuntimeError("Failed to maintain k-regularity during edge swapping")
    
    return G


def _create_base_regular_graph(N, k):
    """Create a base k-regular graph structure."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    # Create k/2 cycles if k is even
    if k % 2 == 0:
        for cycle_num in range(k // 2):
            offset = cycle_num + 1
            for i in range(N):
                next_node = (i + offset) % N
                G.add_edge(i, next_node)
    else:
        # For odd k, create (k-1)/2 cycles plus additional edges
        for cycle_num in range(k // 2):
            offset = cycle_num + 1
            for i in range(N):
                next_node = (i + offset) % N
                G.add_edge(i, next_node)
        
        # Add remaining edges
        for i in range(N):
            if G.degree[i] < k:
                target = (i + N // 2) % N
                if not G.has_edge(i, target) and G.degree[target] < k:
                    G.add_edge(i, target)
    
    return G


def _create_approximate_regular_graph(N, k):
    """Create an approximately k-regular graph as starting point."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    nodes = list(range(N))
    for node in nodes:
        while G.degree[node] < k:
            # Find a node to connect to
            candidates = [n for n in nodes if n != node and 
                         not G.has_edge(node, n) and G.degree[n] < k]
            if not candidates:
                break
            target = random.choice(candidates)
            G.add_edge(node, target)
    
    return G


def _redistribute_edges(G, excess_nodes, deficit_nodes, k):
    """Redistribute edges from excess nodes to deficit nodes."""
    for excess_node in excess_nodes:
        if G.degree[excess_node] <= k:
            continue
            
        neighbors = list(G.neighbors(excess_node))
        for neighbor in neighbors:
            if G.degree[excess_node] <= k:
                break
                
            # Find a deficit node to connect the neighbor to
            for deficit_node in deficit_nodes:
                if (G.degree[deficit_node] < k and 
                    not G.has_edge(neighbor, deficit_node) and
                    deficit_node != neighbor and deficit_node != excess_node):
                    
                    G.remove_edge(excess_node, neighbor)
                    G.add_edge(neighbor, deficit_node)
                    break


def _perform_smart_edge_swap(G, k):
    """Perform an edge swap that moves toward k-regularity."""
    degrees = dict(G.degree())
    excess_nodes = [n for n in G.nodes() if degrees[n] > k]
    deficit_nodes = [n for n in G.nodes() if degrees[n] < k]
    
    if not excess_nodes or not deficit_nodes:
        return _perform_random_edge_swap(G, k)
    
    # Try to swap edges to fix degree issues
    excess_node = random.choice(excess_nodes)
    deficit_node = random.choice(deficit_nodes)
    
    excess_neighbors = list(G.neighbors(excess_node))
    
    for neighbor in excess_neighbors:
        if not G.has_edge(neighbor, deficit_node) and neighbor != deficit_node:
            G.remove_edge(excess_node, neighbor)
            G.add_edge(neighbor, deficit_node)
            return True
    
    return False


def _perform_degree_fixing_swap(G, k):
    """Perform edge operations specifically to fix degree issues."""
    degrees = dict(G.degree())
    deficit_nodes = [n for n in G.nodes() if degrees[n] < k]
    
    if len(deficit_nodes) >= 2:
        # Connect two deficit nodes if possible
        node1 = random.choice(deficit_nodes)
        candidates = [n for n in deficit_nodes if n != node1 and not G.has_edge(node1, n)]
        if candidates:
            node2 = random.choice(candidates)
            G.add_edge(node1, node2)
            return True
    
    return _perform_random_edge_swap(G, k)


def _perform_random_edge_swap(G, k):
    """Perform a random edge swap while maintaining k-regularity."""
    edges = list(G.edges())
    if len(edges) < 2:
        return False
    
    # Select two random edges
    edge1, edge2 = random.sample(edges, 2)
    a, b = edge1
    c, d = edge2
    
    # Check if swap is valid and maintains k-regularity
    if (a != c and a != d and b != c and b != d and
        not G.has_edge(a, c) and not G.has_edge(b, d)):
        
        # Perform the swap
        G.remove_edge(a, b)
        G.remove_edge(c, d)
        G.add_edge(a, c)
        G.add_edge(b, d)
        return True
    
    return False


def _is_k_regular(G, k):
    """Check if graph G is k-regular."""
    return all(G.degree[node] == k for node in G.nodes())


def get_adjacency_matrix(G):
    """
    Get the adjacency matrix of a graph.
    
    Args:
        G (networkx.Graph): Input graph
    
    Returns:
        numpy.ndarray: Adjacency matrix
    """
    return nx.adjacency_matrix(G).toarray()


def generate_multiple_k_regular_graphs(N, k, num_graphs):
    """
    Generate multiple k-regular graphs and return their adjacency matrices.
    
    Args:
        N (int): Number of nodes in each graph
        k (int): Degree of each node
        num_graphs (int): Number of graphs to generate
    
    Returns:
        list: List of adjacency matrices (numpy arrays)
    """
    adjacency_matrices = []
    
    for i in range(num_graphs):
        G = generate_k_regular_graph(N, k)
        adj_matrix = get_adjacency_matrix(G)
        adjacency_matrices.append(adj_matrix)
    
    return adjacency_matrices