# Quantum-like bits

Research software for generating and visualizing quantum-like bits (QL-bits) and networks of coupled QL-bits, based on the following paper:

**Quantum information with quantumlike bits** ([arXiv:2408.06485](https://arxiv.org/abs/2408.06485))

## Installation

```bash
pip install .
```

Dependencies: `numpy`, `scipy`, `networkx`, `matplotlib`.

## Core API

All public functions live in the `core/` package.

```python
from core import generate_k_regular_graph, generate_quantum_like_bit, couple_ql_bits
from core import visualize_ql_bit, visualize_ql_network, visualize_ql_network_ring, visualize_ql_network_circular
from core import compute_eigenspectrum, spectral_density_exact, spectral_density_approx
from core import plot_eigenvalue_histogram, plot_spectral_density
from core import minimal_quotient, contract_ql_bit
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

---

### Eigenspectral analysis

Two computation paths are provided. Both split the spectrum into **emergent (coherent) states** — the top eigenvalues, kept as sharp features — and the **incoherent bulk**, which is Gaussian-smoothed.

For an n-bit QL-bit network the expected number of emergent states is `2**n_bits`.

#### Eigendecomposition

```python
eigenvalues, eigenvectors = compute_eigenspectrum(adj_matrix)
```

Uses `numpy.linalg.eigh` (symmetric-aware, real eigenvalues guaranteed). Returns both arrays sorted by `|λ|` descending.

#### Exact spectral density

Computes the density of states directly from the full eigendecomposition. Precise but scales as O(N³) — practical up to moderate matrix sizes.

```python
x, rho_bulk, rho_emergent = spectral_density_exact(
    adj_matrix,
    num_emergent=2**n_bits,  # e.g. 4 for a 2-bit network
    x_range=(-30, 100),      # eigenvalue axis range; auto-derived if None
    bins=8000,
    sigma=50.0,              # Gaussian smoothing width in bins for bulk
)
```

#### Approximate spectral density (convolution)

Approximates the spectrum of an n-bit Cartesian-product network by convolving the single-bit spectrum `n_bits` times. Avoids constructing the full coupled matrix, making it practical for large `n_bits`.

The product rule applied at each convolution step keeps emergent states from contaminating the smooth bulk hump:
- `new_emergent = emergent ⊛ emergent`
- `new_bulk = (bulk ⊛ bulk) + (bulk ⊛ emergent) + (emergent ⊛ bulk)`

```python
x, rho_bulk, rho_emergent = spectral_density_approx(
    ql_adj_matrix,           # single QL-bit matrix (the base unit)
    n_bits=3,
    x_range=(-30, 100),      # range for the single-bit spectrum; auto-derived if None
    bins=8000,
    sigma=50.0,
)
```

---

### Spectral visualization

#### All eigenstates — histogram

```python
plot_eigenvalue_histogram(eigenvalues, x_range=(-30, 100), bins=200)
```

Plots a raw histogram of all eigenvalues with no emergent/bulk separation.

#### Split density — emergent lines + smooth bulk

Takes the three arrays returned by either `spectral_density_exact` or `spectral_density_approx`.  Emergent states are drawn as vertical lines; the incoherent bulk as a smooth continuous curve on a log y-axis.

```python
plot_spectral_density(
    x, rho_bulk, rho_emergent,
    x_range=(-30, 100),
    y_range=(1e-6, 10),
    color='forestgreen',
    label='My spectrum',       # optional legend label
    log_scale=True,
    show_plot=True,            # display interactively
    save_path='spectrum.png',  # or omit to use default filename
)
```

#### Full example

```python
from core import generate_quantum_like_bit, spectral_density_approx, plot_spectral_density

ql, info = generate_quantum_like_bit(N=40, k=20, l=1)

# Approximate spectrum for a 3-bit network without building the coupled matrix
x, rho_bulk, rho_emergent = spectral_density_approx(ql, n_bits=3, x_range=(-30, 100))

plot_spectral_density(x, rho_bulk, rho_emergent, x_range=(-90, 300), show_plot=True)
```

---

### Graph contraction (equitable partition / minimal quotient)

The Cartesian product of N QL-bits grows as `(2N)^N` nodes, which becomes intractable quickly. An equitable partition that groups nodes by subgraph membership within each constituent QL-bit yields a **divisor (quotient) matrix** that preserves all `2^N` emergent eigenvalues while discarding the bulk incoherent states. This is Theorem 1 of Horvat et al.

The emergent eigenvalues of the quotient are all `2^N` sign combinations:
```
λ = (Σ k_q) ± l_1 ± l_2 ± ... ± l_N
```

Two functions are provided depending on whether the full Cartesian product matrix is available.

#### Direct construction — no Cartesian product required

Builds the `2^N × 2^N` minimal quotient matrix directly from the `(k, l)` parameters of each QL-bit. This is the preferred path for large systems where forming the full product matrix is impractical.

```python
Q, q_info = minimal_quotient(ql1, ql2)          # 4×4 for two QL-bits
Q, q_info = minimal_quotient(ql1, ql2, ql3)     # 8×8 for three QL-bits
```

- `Q` — the `2^N × 2^N` divisor matrix
- `q_info['diagonal_shift']` — the `Σ k_q` self-loop contribution on the diagonal (a global spectral shift; remove it to centre eigenvalues at `± l_q`)

#### Iterative contraction from the full matrix

Given the output of `couple_ql_bits`, `contract_ql_bit` collapses one QL-bit at a time to its 2-node quotient using the characteristic matrix formula `Aπ = (1/N_q) · S^T A S`. Call it repeatedly in any order — the result is independent of the sequence (Lemma 4, associativity of fibrations).

```python
# Contract ql2 first (target index 1) → intermediate 2·n1 × 2·n1 matrix
mid, mid_info = contract_ql_bit((coupled, net_info), target=1)

# Contract ql1 next (target index 0) → full 4×4 minimal quotient
full_min, fm_info = contract_ql_bit((mid, mid_info), target=0)
```

`target` is the 0-based index into `net_info['ql_bits']`, where index `0` is the innermost QL-bit (first argument to `couple_ql_bits`) and index `N-1` is the outermost.

The intermediate matrix after one contraction is itself a valid (smaller) QL-bit network and can be passed directly to `compute_eigenspectrum` or the spectral density functions.

#### Full example

```python
from core import (
    generate_quantum_like_bit, couple_ql_bits,
    minimal_quotient, contract_ql_bit,
    compute_eigenspectrum,
)

ql1 = generate_quantum_like_bit(N=20, k=12, l=1)
ql2 = generate_quantum_like_bit(N=20, k=12, l=1)

# Full Cartesian product (1600×1600)
coupled, net_info = couple_ql_bits(ql1, ql2)
evals_full, _ = compute_eigenspectrum(coupled)

# Intermediate: contract ql2 only (80×80, preserves ql1 at full size)
mid, mid_info = contract_ql_bit((coupled, net_info), target=1)
evals_mid, _ = compute_eigenspectrum(mid)

# Full minimal quotient — two equivalent routes:
full_min, _ = contract_ql_bit((mid, mid_info), target=0)   # iterative
Q_direct, _ = minimal_quotient(ql1, ql2)                   # direct (no full matrix needed)

# All three give the same 4 emergent eigenvalues: k1+k2 ± l1 ± l2
print(compute_eigenspectrum(full_min)[0].round(4))
print(compute_eigenspectrum(Q_direct)[0].round(4))
```
