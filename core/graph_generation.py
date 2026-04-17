"""
graph_generation.py

Generate k-regular graphs, quantum-like bits (QL-bits), and networks of QL-bits.

A QL-bit is two identically-sized k-regular subgraphs coupled together
with l cross-edges per node.

A QL-bit network couples two or more QL-bits via the graph Cartesian product:
G □ H places a copy of G at every node of H, connecting node i in copy u
to node i in copy v whenever u ~ v in H.

Public API
----------
generate_k_regular_graph(N, k)          -> np.ndarray  (N x N adjacency matrix)
generate_quantum_like_bit(N, k, l)      -> (np.ndarray, dict)
couple_ql_bits(ql_bit_a, ql_bit_b, ...) -> (np.ndarray, dict)
"""

import random
import warnings
import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def generate_k_regular_graph(N, k):
    """
    Generate a random k-regular graph on N nodes.

    Parameters
    ----------
    N : int  Number of nodes
    k : int  Degree of every node

    Returns
    -------
    np.ndarray  N x N integer adjacency matrix

    Raises
    ------
    ValueError  If the parameters cannot produce a valid k-regular graph
    RuntimeError  If generation fails
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k >= N:
        raise ValueError(f"k ({k}) must be less than N ({N})")
    if (N * k) % 2 != 0:
        raise ValueError(f"N*k ({N*k}) must be even for a valid k-regular graph")

    if k == 0:
        G = nx.empty_graph(N)
    elif k == N - 1:
        G = nx.complete_graph(N)
    else:
        density = k / (N - 1)
        if density > 0.7:
            G = _by_edge_removal(N, k)
        elif k <= 3 or (k <= 6 and N <= 50):
            G = _by_incremental(N, k)
        else:
            G = _by_configuration(N, k)

    return nx.adjacency_matrix(G).toarray().astype(int)


def generate_quantum_like_bit(N, k, l):
    """
    Generate a quantum-like bit: two k-regular subgraphs on N nodes each,
    coupled with l cross-edges per node.

    Parameters
    ----------
    N : int  Number of nodes in each subgraph
    k : int  Regularity degree of each subgraph
    l : int  Number of coupling edges per node (1 <= l < N)

    Returns
    -------
    adjacency_matrix : np.ndarray  (2N x 2N) combined adjacency matrix.
        Rows/columns 0..N-1 are subgraph 1 (a-nodes);
        rows/columns N..2N-1 are subgraph 2 (b-nodes).
    graph_info : dict  with keys N, k, l, total_nodes, total_edges, coupling_edges

    Raises
    ------
    ValueError  If parameters are invalid
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k >= N:
        raise ValueError(f"k ({k}) must be less than N ({N})")
    if (N * k) % 2 != 0:
        raise ValueError(f"N*k ({N*k}) must be even for a valid k-regular graph")
    if not (1 <= l <= N):
        raise ValueError(f"l ({l}) must satisfy 1 <= l <= N ({N})")

    adj1 = generate_k_regular_graph(N, k)
    adj2 = generate_k_regular_graph(N, k)
    coupling = _balanced_coupling(N, l)

    matrix = np.zeros((2 * N, 2 * N), dtype=int)
    matrix[:N, :N] = adj1
    matrix[N:, N:] = adj2
    matrix[:N, N:] = coupling
    matrix[N:, :N] = coupling.T

    graph_info = {
        'N': N,
        'k': k,
        'l': l,
        'total_nodes': 2 * N,
        'total_edges': int(np.sum(matrix) // 2),
        'coupling_edges': int(np.sum(coupling)),
    }

    return matrix, graph_info


def couple_ql_bits(*ql_bits):
    """
    Couple two or more QL-bits via the graph Cartesian product.

    The Cartesian product G □ H places a copy of G at every node of H.
    Node i in copy u connects to node i in copy v whenever u ~ v in H.
    For three or more QL-bits the product is chained left-to-right:
    (QL1 □ QL2) □ QL3 □ ...

    Parameters
    ----------
    *ql_bits : two or more (np.ndarray, dict) tuples
        Each is the direct output of generate_quantum_like_bit (or a prior
        call to couple_ql_bits).

    Returns
    -------
    matrix : np.ndarray   Combined adjacency matrix
    network_info : dict   Keys: num_ql_bits, total_nodes, total_edges, ql_bits (list of infos)

    Raises
    ------
    ValueError  If fewer than two QL-bits are provided
    """
    if len(ql_bits) < 2:
        raise ValueError("couple_ql_bits requires at least two QL-bits")

    result = _cartesian_product_pair(ql_bits[0], ql_bits[1])
    for ql_bit in ql_bits[2:]:
        result = _cartesian_product_pair(result, ql_bit)
    return result


def _cartesian_product_pair(ql_a, ql_b):
    """
    Pairwise Cartesian product.
    A(G □ H) = I_|H| ⊗ A(G) + A(H) ⊗ I_|G|

    Node index i = h * n_a + g, where h is the copy (H-index) and g is the
    position within the copy (G-index).  This keeps each copy's nodes
    consecutive: copy h owns nodes [h*n_a … (h+1)*n_a - 1].
    """
    matrix_a, info_a = ql_a
    matrix_b, info_b = ql_b
    n_a, n_b = matrix_a.shape[0], matrix_b.shape[0]
    n_total  = n_a * n_b

    estimated_mb = (n_total ** 2 * 8) / 1024 / 1024
    if estimated_mb > 512:
        warnings.warn(
            f"Cartesian product will produce a {n_total}×{n_total} matrix "
            f"(~{estimated_mb/1024:.1f} GB). Consider smaller N.",
            stacklevel=3,
        )

    matrix = (np.kron(np.eye(n_b, dtype=int), matrix_a) +
              np.kron(matrix_b, np.eye(n_a, dtype=int)))

    # Flatten nested ql_bit info lists so chaining stays readable
    infos_a = info_a.get('ql_bits', [info_a])
    infos_b = info_b.get('ql_bits', [info_b])
    all_infos = infos_a + infos_b

    network_info = {
        'num_ql_bits': len(all_infos),
        'total_nodes': matrix.shape[0],
        'total_edges': int(np.sum(matrix) // 2),
        'ql_bits': all_infos,
    }
    return matrix, network_info


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------

def _balanced_coupling(N, l):
    """Round-robin coupling: node i connects to nodes (i+1) % N … (i+l) % N."""
    coupling = np.zeros((N, N), dtype=int)
    for i in range(N):
        for offset in range(1, l + 1):
            coupling[i, (i + offset) % N] = 1
    return coupling


# ---------------------------------------------------------------------------
# k-regular graph generation algorithms (private)
# ---------------------------------------------------------------------------

def _by_configuration(N, k):
    """Configuration model with multi-edge / self-loop cleaning."""
    for _ in range(50):
        try:
            G = nx.configuration_model([k] * N, seed=random.randint(0, 1_000_000))
            G = _clean_configuration(G, k)
            if G is not None and _is_k_regular(G, k):
                return G
        except Exception:
            continue
    return _by_edge_swapping(N, k)


def _clean_configuration(G, k):
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G = nx.Graph(G)

    for _ in range(200):
        if _is_k_regular(G, k):
            break
        degrees = dict(G.degree())
        deficit = [n for n in G if degrees[n] < k]
        excess  = [n for n in G if degrees[n] > k]

        if len(deficit) >= 2:
            random.shuffle(deficit)
            for i in range(0, len(deficit) - 1, 2):
                u, v = deficit[i], deficit[i + 1]
                if not G.has_edge(u, v) and degrees[u] < k and degrees[v] < k:
                    G.add_edge(u, v)
                    degrees[u] += 1
                    degrees[v] += 1
        elif excess and deficit:
            _redistribute(G, excess, deficit, k)
        else:
            _random_swap(G)

    return G if _is_k_regular(G, k) else None


def _by_edge_removal(N, k):
    """Start from K_N and remove edges until k-regular."""
    G = nx.complete_graph(N)
    to_remove = G.number_of_edges() - (N * k) // 2

    edges = list(G.edges())
    random.shuffle(edges)
    removed = 0
    for u, v in edges:
        if removed >= to_remove:
            break
        if G.degree[u] > k and G.degree[v] > k:
            G.remove_edge(u, v)
            removed += 1

    for _ in range(N * 10):
        if _is_k_regular(G, k):
            break
        _smart_swap(G, k)

    if not _is_k_regular(G, k):
        raise RuntimeError(f"Edge-removal failed to produce a {k}-regular graph on {N} nodes")
    return G


def _by_incremental(N, k):
    """Greedy incremental construction, suitable for low-degree graphs."""
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for node in range(N):
        needed = k - G.degree[node]
        candidates = sorted(
            [n for n in range(N)
             if n != node and not G.has_edge(node, n) and G.degree[n] < k],
            key=lambda n: G.degree[n],
        )
        for c in candidates[:needed]:
            if G.degree[node] < k:
                G.add_edge(node, c)

    for _ in range(N * 50):
        if _is_k_regular(G, k):
            break
        _degree_fixing_swap(G, k)

    if not _is_k_regular(G, k):
        raise RuntimeError(f"Incremental construction failed for {k}-regular graph on {N} nodes")
    return G


def _by_edge_swapping(N, k):
    """Build a structured starting graph then randomise via edge swaps."""
    if k == 2:
        G = nx.cycle_graph(N)
    elif k % 2 == 0 and k <= N // 2:
        G = _base_regular(N, k)
    else:
        G = _approx_regular(N, k)

    target  = min(N * k * 2, 10_000)
    cap     = min(target * 3, 20_000)
    done = attempts = 0
    while done < target and attempts < cap:
        attempts += 1
        if _random_swap(G):
            done += 1

    if not _is_k_regular(G, k):
        raise RuntimeError(f"Edge-swapping failed to produce a {k}-regular graph on {N} nodes")
    return G


def _base_regular(N, k):
    """Deterministic k-regular starting graph (union of offset cycles)."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for c in range(k // 2):
        offset = c + 1
        for i in range(N):
            G.add_edge(i, (i + offset) % N)
    if k % 2 == 1:
        for i in range(N):
            t = (i + N // 2) % N
            if not G.has_edge(i, t) and G.degree[i] < k and G.degree[t] < k:
                G.add_edge(i, t)
    return G


def _approx_regular(N, k):
    """Greedy approximate k-regular graph as a swap starting point."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node in range(N):
        while G.degree[node] < k:
            candidates = [n for n in range(N)
                          if n != node and not G.has_edge(node, n) and G.degree[n] < k]
            if not candidates:
                break
            G.add_edge(node, random.choice(candidates))
    return G


# ---------------------------------------------------------------------------
# Edge-manipulation helpers
# ---------------------------------------------------------------------------

def _redistribute(G, excess, deficit, k):
    for e in excess:
        if G.degree[e] <= k:
            continue
        for nbr in list(G.neighbors(e)):
            if G.degree[e] <= k:
                break
            for d in deficit:
                if G.degree[d] < k and not G.has_edge(nbr, d) and d != nbr and d != e:
                    G.remove_edge(e, nbr)
                    G.add_edge(nbr, d)
                    break


def _smart_swap(G, k):
    degrees = dict(G.degree())
    excess  = [n for n in G if degrees[n] > k]
    deficit = [n for n in G if degrees[n] < k]
    if not excess or not deficit:
        return _random_swap(G)
    e, d = random.choice(excess), random.choice(deficit)
    for nbr in list(G.neighbors(e)):
        if not G.has_edge(nbr, d) and nbr != d:
            G.remove_edge(e, nbr)
            G.add_edge(nbr, d)
            return True
    return False


def _degree_fixing_swap(G, k):
    deficit = [n for n in G if G.degree[n] < k]
    if len(deficit) >= 2:
        u = random.choice(deficit)
        candidates = [n for n in deficit if n != u and not G.has_edge(u, n)]
        if candidates:
            G.add_edge(u, random.choice(candidates))
            return True
    return _random_swap(G)


def _random_swap(G):
    """Degree-preserving random edge swap."""
    edges = list(G.edges())
    if len(edges) < 2:
        return False
    (a, b), (c, d) = random.sample(edges, 2)
    if len({a, b, c, d}) == 4 and not G.has_edge(a, c) and not G.has_edge(b, d):
        G.remove_edge(a, b)
        G.remove_edge(c, d)
        G.add_edge(a, c)
        G.add_edge(b, d)
        return True
    return False


def _is_k_regular(G, k):
    return all(G.degree[n] == k for n in G)
