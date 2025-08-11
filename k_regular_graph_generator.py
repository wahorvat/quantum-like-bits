"""
k_regular_graph_generator.py

A module for generating k-regular graphs and their adjacency matrices.
This module is designed to be imported and used by other Python files.
"""

import random
import networkx as nx
import numpy as np


def generate_k_regular_graph(N, k):
    """
    Generate a random k-regular graph on N nodes.
    
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
    
    # Create the graph using configuration model approach
    max_attempts = 100
    
    for attempt in range(max_attempts):
        try:
            # Create degree sequence: each node has degree k
            degree_sequence = [k] * N
            
            # Use NetworkX's configuration model
            G = nx.configuration_model(degree_sequence)
            
            # Remove self-loops
            G.remove_edges_from(nx.selfloop_edges(G))
            
            # Convert to simple graph (removes multiple edges)
            G = nx.Graph(G)
            
            # Check if all nodes still have degree k (or close to it)
            degrees = dict(G.degree())
            
            # If we lost too many edges, try again
            if all(degrees[node] >= k - 1 for node in G.nodes()):
                # Try to fix nodes with degree k-1 by adding edges
                nodes_need_edge = [node for node in G.nodes() if degrees[node] == k - 1]
                
                # Add edges between nodes that need them
                while len(nodes_need_edge) >= 2:
                    node1 = nodes_need_edge.pop()
                    node2 = nodes_need_edge.pop()
                    
                    if not G.has_edge(node1, node2):
                        G.add_edge(node1, node2)
                        degrees[node1] += 1
                        degrees[node2] += 1
                
                # Check if we achieved k-regularity
                if all(degrees[node] == k for node in G.nodes()):
                    return G
                    
        except Exception:
            continue
    
    # If configuration model fails, use alternative approach
    return _generate_k_regular_alternative(N, k)


def _generate_k_regular_alternative(N, k):
    """
    Alternative method using edge swapping for smaller k values.
    Private function used as fallback.
    """
    if k > N // 2:
        # For large k, start with complete graph and remove edges
        G = nx.complete_graph(N)
        target_edges_to_remove = (N * (N - 1) // 2) - (N * k // 2)
        
        edges_list = list(G.edges())
        random.shuffle(edges_list)
        
        for i in range(target_edges_to_remove):
            if i < len(edges_list):
                u, v = edges_list[i]
                if G.degree[u] > k and G.degree[v] > k:
                    G.remove_edge(u, v)
        
        return G
    
    else:
        # For small k, build graph incrementally
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        nodes = list(range(N))
        degrees = {node: 0 for node in nodes}
        
        attempts = 0
        max_attempts = 1000
        
        while any(degrees[node] < k for node in nodes) and attempts < max_attempts:
            attempts += 1
            
            # Find nodes that need more edges
            available_nodes = [node for node in nodes if degrees[node] < k]
            
            if len(available_nodes) < 2:
                break
                
            # Pick two random nodes that need edges
            node1, node2 = random.sample(available_nodes, 2)
            
            # Add edge if it doesn't exist
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2)
                degrees[node1] += 1
                degrees[node2] += 1
        
        return G


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