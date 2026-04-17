# Quantum-like bits

Research software for generating and visualizing quantum-like bits (QL-bits) and networks of coupled QL-bits, based on the following paper:

**Quantum information with quantumlike bits** ([arXiv:2408.06485](https://arxiv.org/abs/2408.06485))

## Installation

```bash
pip install .
```

Dependencies: `numpy`, `networkx`, `matplotlib`.

## Core API

All public functions live in the `core/` package.

```python
from core import generate_k_regular_graph, generate_quantum_like_bit, couple_ql_bits
from core import visualize_ql_bit, visualize_ql_network, visualize_ql_network_ring, visualize_ql_network_circular
```

---

### Generating a k-regular graph

```python
adj = generate_k_regular_graph(N, k)
```

Returns an `N x N` integer adjacency matrix for a random k-regular graph.

---

### Generating a QL-bit

A QL-bit is two k-regular subgraphs on N nodes each, coupled with `l` cross-edges per node.

```python
matrix, info = generate_quantum_like_bit(N, k, l)
```

- `matrix` — `(2N x 2N)` adjacency matrix; rows/columns `0..N-1` are subgraph 1, `N..2N-1` are subgraph 2
- `info` — dict with keys `N`, `k`, `l`, `total_nodes`, `total_edges`, `coupling_edges`

```python
# Example: 6-node subgraphs, 4-regular, 1 coupling edge per node
matrix, info = generate_quantum_like_bit(6, 4, 1)

# Minimal QL-bit: single node per subgraph
matrix, info = generate_quantum_like_bit(1, 0, 1)
```

---

### Coupling QL-bits (Cartesian product)

Two or more QL-bits can be coupled via the graph Cartesian product `G □ H`, which places a copy of G at every node of H.

```python
network_matrix, network_info = couple_ql_bits(ql_bit_a, ql_bit_b)
network_matrix, network_info = couple_ql_bits(ql_bit_a, ql_bit_b, ql_bit_c, ...)
```

For three or more QL-bits the product chains left-to-right: `(A □ B) □ C □ ...`

```python
ql1 = generate_quantum_like_bit(6, 4, 1)
ql2 = generate_quantum_like_bit(3, 2, 1)

# 2 QL-bit network
net, net_info = couple_ql_bits(ql1, ql2)

# Chain a third
ql3 = generate_quantum_like_bit(1, 0, 1)
net2, net2_info = couple_ql_bits((net, net_info), ql3)
```

---

### Visualization

All visualization functions save a PNG by default (`show_plot=False`). Pass `show_plot=True` to display interactively, or `save_path='my_file.png'` to set the output path explicitly.

**Single QL-bit** — subgraph 1 in lightcoral, subgraph 2 in skyblue, coupling edges in gray:

```python
visualize_ql_bit(matrix, info)
visualize_ql_bit(matrix, info, show_plot=True)
```

**QL-bit network** — three layout options; each copy of the base QL-bit is a distinct color:

```python
# Spring layout (auto-falls back to circular above 200 nodes)
visualize_ql_network(network_matrix, network_info)

# Copies arranged in a ring, nodes clustered per copy
visualize_ql_network_ring(network_matrix, network_info)

# All nodes on a single circle (nx.circular_layout)
visualize_ql_network_circular(network_matrix, network_info)
```

Edge types in network visualizations:
- **Colored solid** — intra-copy edges (color matches copy)
- **Dashed gray** — inter-copy edges added by the Cartesian product
- **Solid dark gray** — intra-QL-bit coupling edges

<!-- ---

### Eigenvalue analysis

`eigenvalue_analysis.py` provides standalone utilities for computing and comparing eigenvalue spectra of QL-bit structures.

--- -->

<!-- ## Notebooks

| Notebook | Contents |
|---|---|
| `scratch.ipynb` | General exploration |
| `entangled two QL viz.ipynb` | Two-QL-bit visualization examples |
| `graph_spectra_viz.ipynb` | Spectral analysis of QL-bit graphs |
| `computational_basis_poc.ipynb` | Computational basis proof-of-concept | -->
