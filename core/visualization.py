"""
visualization.py

Plotting utilities for quantum-like bits and QL-bit networks.

By default every function saves a PNG to disk (show_plot=False).
Pass show_plot=True to display interactively instead.
Pass save_path='my_file.png' to choose the output path explicitly;
omit it to use an auto-generated name in the current directory.

Public API
----------
visualize_ql_bit(ql_matrix, graph_info, show_plot=False, save_path=None)
visualize_ql_network(network_matrix, network_info, show_plot=False, save_path=None)
visualize_ql_network_ring(network_matrix, network_info, show_plot=False, save_path=None)
visualize_ql_network_circular(network_matrix, network_info, show_plot=False, save_path=None)

Performance notes
-----------------
Spring layout (visualize_ql_network) runs O(n²) force calculations per
iteration and becomes slow above ~200 nodes.  For larger networks the
function automatically falls back to circular layout.

The Cartesian product matrix grows as O((2N)^(2 * num_ql_bits)).
For two QL-bits with N=20 the matrix is 1600×1600 (~20 MB, fast).
For three QL-bits with N=20 it is 64000×64000 (~32 GB, impractical).
Keep N small when coupling three or more QL-bits.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import networkx as nx


# ---------------------------------------------------------------------------
# Color palettes  (from project notebooks — entangled two QL viz.ipynb)
# ---------------------------------------------------------------------------

_COLOR_A = 'lightcoral'   # subgraph 1 of a single QL-bit
_COLOR_B = 'skyblue'      # subgraph 2 of a single QL-bit

_COPY_COLORS = [
    'lightcoral',
    '#77B1F6',
    '#EAAC38',
    '#4CA889',
    '#EA3368',
    '#7A339C',
    '#78AA61',
    '#F7CD46',
    '#67ABE0',
    '#DCA237',
    '#C76526',
    '#8761CD',
]

_COLOR_COUPLING = 'gray'
_COLOR_INTER    = 'gray'
_COLOR_INTRA_C  = 'dimgray'

# Spring layout is replaced by circular above this node count
_SPRING_LAYOUT_LIMIT = 200


# ---------------------------------------------------------------------------
# Single QL-bit
# ---------------------------------------------------------------------------

def visualize_ql_bit(ql_matrix, graph_info, show_plot=False, save_path=None):
    """
    Visualise a single quantum-like bit.

    Subgraph 1 (a-nodes) in lightcoral on the left;
    subgraph 2 (b-nodes) in skyblue on the right;
    coupling edges in gray.

    Parameters
    ----------
    ql_matrix  : np.ndarray   (2N x 2N) adjacency matrix
    graph_info : dict         Output of generate_quantum_like_bit
    show_plot  : bool         Display interactively (default False)
    save_path  : str | None   PNG path; auto-generated if None

    Returns
    -------
    matplotlib.figure.Figure
    """
    N = graph_info['N']
    k = graph_info['k']
    l = graph_info['l']
    coupling_edges = graph_info['coupling_edges']

    G = nx.from_numpy_array(ql_matrix)
    a_nodes = list(range(N))
    b_nodes = list(range(N, 2 * N))

    if N <= 6:
        fig_size = (12, 6);  node_size = 600;  font_size = 12;  edge_w = 6;  sep = 2.5
    elif N <= 15:
        fig_size = (15, 8);  node_size = 400;  font_size = 10;  edge_w = 4;  sep = 3.0
    else:
        fig_size = (18, 10); node_size = 300;  font_size = 8;   edge_w = 3;  sep = 3.5

    fig, ax = plt.subplots(figsize=fig_size)

    if N == 1:
        pos = {0: (-sep, 0), 1: (sep, 0)}
    else:
        pos = {}
        for node, (x, y) in nx.spring_layout(G.subgraph(a_nodes), seed=42).items():
            pos[node] = (x - sep, y)
        for node, (x, y) in nx.spring_layout(G.subgraph(b_nodes), seed=43).items():
            pos[node] = (x + sep, y)

    internal_a = [(u, v) for u, v in G.edges() if u in a_nodes and v in a_nodes]
    internal_b = [(u, v) for u, v in G.edges() if u in b_nodes and v in b_nodes]
    cross       = [(u, v) for u, v in G.edges() if (u in a_nodes) != (v in a_nodes)]

    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, edge_color='lightgray', ax=ax)
    if internal_a:
        nx.draw_networkx_edges(G, pos, edgelist=internal_a,
                               width=edge_w, alpha=0.7, edge_color=_COLOR_A, ax=ax)
    if internal_b:
        nx.draw_networkx_edges(G, pos, edgelist=internal_b,
                               width=edge_w, alpha=0.7, edge_color=_COLOR_B, ax=ax)
    if cross:
        nx.draw_networkx_edges(G, pos, edgelist=cross,
                               width=2, alpha=0.8, edge_color=_COLOR_COUPLING, ax=ax)

    node_style = {'edgecolors': 'dimgray', 'alpha': 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=a_nodes,
                           node_color=_COLOR_A, node_size=node_size, ax=ax, **node_style)
    nx.draw_networkx_nodes(G, pos, nodelist=b_nodes,
                           node_color=_COLOR_B, node_size=node_size, ax=ax, **node_style)

    labels = {i: f'$a_{{{i}}}$' for i in range(N)}
    labels.update({N + i: f'$b_{{{i}}}$' for i in range(N)})
    nx.draw_networkx_labels(G, pos, labels,
                            font_size=font_size, font_color='white',
                            font_weight='bold', ax=ax)

    ax.set_title(
        f"Quantum-Like Bit — {k}-regular, N={N} nodes per subgraph, l={l} coupling per node",
        fontsize=14, fontweight='bold', pad=20,
    )
    ax.axis('off')

    legend = [
        Patch(facecolor=_COLOR_A, edgecolor='dimgray',
              label=f'Subgraph 1  ({N} a-nodes, {k}-regular)'),
        Patch(facecolor=_COLOR_B, edgecolor='dimgray',
              label=f'Subgraph 2  ({N} b-nodes, {k}-regular)'),
        mlines.Line2D([], [], color=_COLOR_A, linewidth=edge_w,
                      label='Internal edges — subgraph 1'),
        mlines.Line2D([], [], color=_COLOR_B, linewidth=edge_w,
                      label='Internal edges — subgraph 2'),
        mlines.Line2D([], [], color=_COLOR_COUPLING, linewidth=2,
                      label=f'Coupling edges  ({coupling_edges} total)'),
    ]
    ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = f'ql_bit_N{N}_k{k}_l{l}.png'
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# QL-bit networks — shared drawing core
# ---------------------------------------------------------------------------

def _draw_network(ax, G, pos, n_total, n_copies, n_per_copy, n_half, colors,
                  node_size, edge_w, font_size):
    inter, intra, coupling = _classify_edges(G, n_copies, n_per_copy, n_half)

    if inter:
        nx.draw_networkx_edges(G, pos, edgelist=inter,
                               width=0.8, alpha=0.3, edge_color=_COLOR_INTER,
                               style='dashed', ax=ax)
    if coupling:
        nx.draw_networkx_edges(G, pos, edgelist=coupling,
                               width=1.0, alpha=0.6, edge_color=_COLOR_INTRA_C, ax=ax)
    for h in range(n_copies):
        if intra[h]:
            nx.draw_networkx_edges(G, pos, edgelist=intra[h],
                                   width=edge_w, alpha=0.7, edge_color=colors[h], ax=ax)

    for h in range(n_copies):
        nodes = list(range(h * n_per_copy, (h + 1) * n_per_copy))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                               node_color=colors[h], node_size=node_size,
                               edgecolors='dimgray', alpha=0.9, ax=ax)

    if n_total <= 36:
        nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(n_total)},
                                font_size=font_size, font_color='white',
                                font_weight='bold', ax=ax)


# ---------------------------------------------------------------------------
# QL-bit networks (spring layout)
# ---------------------------------------------------------------------------

def visualize_ql_network(network_matrix, network_info, show_plot=False, save_path=None):
    """
    Spring-layout visualisation of a coupled QL-bit network.

    Automatically falls back to circular layout when the network has more
    than 200 nodes, where spring layout becomes prohibitively slow.

    Parameters
    ----------
    network_matrix : np.ndarray   Combined adjacency matrix from couple_ql_bits
    network_info   : dict         Output dict from couple_ql_bits
    show_plot      : bool         Display interactively (default False)
    save_path      : str | None   PNG path; auto-generated if None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_total, n_copies, n_per_copy, n_half, colors = _parse_network(network_info)
    G = nx.from_numpy_array(network_matrix)

    if n_total <= _SPRING_LAYOUT_LIMIT:
        main_r    = max(2.0, n_copies * 0.6)
        cluster_r = min(0.5, main_r / (n_copies + 1))
        init_pos  = {}
        for i in range(n_total):
            h, g   = i // n_per_copy, i % n_per_copy
            angle  = 2 * np.pi * h / n_copies
            cx, cy = main_r * np.cos(angle), main_r * np.sin(angle)
            inner  = 2 * np.pi * g / n_per_copy
            init_pos[i] = (cx + cluster_r * np.cos(inner), cy + cluster_r * np.sin(inner))
        iterations = max(30, 80 - n_total // 5)
        pos = nx.spring_layout(G, pos=init_pos, k=0.3, iterations=iterations, seed=42)
    else:
        pos = nx.circular_layout(G)

    node_size = max(30,  400 - n_total * 3)
    edge_w    = max(0.5, 3.0 - n_copies / 5)
    font_size = max(6,   10  - n_total // 15)

    fig, ax = plt.subplots(figsize=(14, 10))
    _draw_network(ax, G, pos, n_total, n_copies, n_per_copy, n_half, colors,
                  node_size, edge_w, font_size)

    ax.legend(handles=_network_legend(n_copies, colors),
              loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)
    ax.set_title(
        f"QL-bit Network — {network_info['num_ql_bits']} QL-bits  |  "
        f"{n_copies} copies  |  {n_total} total nodes"
        + ('' if n_total <= _SPRING_LAYOUT_LIMIT else '  [circular fallback]'),
        fontsize=13, fontweight='bold',
    )
    ax.axis('off')
    plt.tight_layout()

    if save_path is None:
        save_path = f"ql_network_{network_info['num_ql_bits']}bits_{n_total}nodes_spring.png"
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# QL-bit networks (ring layout)
# ---------------------------------------------------------------------------

def visualize_ql_network_ring(network_matrix, network_info, show_plot=False, save_path=None):
    """
    Ring-layout visualisation of a coupled QL-bit network.

    Each copy occupies an equal arc; nodes within each arc are arranged
    in a small circle.

    Parameters
    ----------
    network_matrix : np.ndarray   Combined adjacency matrix from couple_ql_bits
    network_info   : dict         Output dict from couple_ql_bits
    show_plot      : bool         Display interactively (default False)
    save_path      : str | None   PNG path; auto-generated if None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_total, n_copies, n_per_copy, n_half, colors = _parse_network(network_info)
    G = nx.from_numpy_array(network_matrix)

    main_r    = max(3.0, n_copies * 0.9)
    cluster_r = min(0.8, main_r / (n_copies + 1))

    pos = {}
    for h in range(n_copies):
        angle  = 2 * np.pi * h / n_copies - np.pi / 2
        cx, cy = main_r * np.cos(angle), main_r * np.sin(angle)
        nodes  = list(range(h * n_per_copy, (h + 1) * n_per_copy))
        _place_cluster(pos, nodes, cx, cy, cluster_r)

    node_size = max(20,  300 - n_total * 2)
    edge_w    = max(0.5, 3.0 - n_copies / 5)
    font_size = max(6,   10  - n_total // 20)
    fig_dim   = max(10, main_r * 2 + 3)

    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim))
    _draw_network(ax, G, pos, n_total, n_copies, n_per_copy, n_half, colors,
                  node_size, edge_w, font_size)

    for h in range(n_copies):
        angle  = 2 * np.pi * h / n_copies - np.pi / 2
        cx, cy = main_r * np.cos(angle), main_r * np.sin(angle)
        ax.text(cx, cy, f'C{h}', fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    ax.legend(handles=_network_legend(n_copies, colors),
              loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)
    ax.set_title(
        f"QL-bit Network Ring — {network_info['num_ql_bits']} QL-bits  |  "
        f"{n_copies} copies  |  {n_total} total nodes",
        fontsize=13, fontweight='bold',
    )
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path is None:
        save_path = f"ql_network_{network_info['num_ql_bits']}bits_{n_total}nodes_ring.png"
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# QL-bit networks (NetworkX circular layout)
# ---------------------------------------------------------------------------

def visualize_ql_network_circular(network_matrix, network_info, show_plot=False, save_path=None):
    """
    Circular-layout visualisation using nx.circular_layout.

    All nodes placed on a single circle in index order.
    Copy membership shown through node colour.

    Parameters
    ----------
    network_matrix : np.ndarray   Combined adjacency matrix from couple_ql_bits
    network_info   : dict         Output dict from couple_ql_bits
    show_plot      : bool         Display interactively (default False)
    save_path      : str | None   PNG path; auto-generated if None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_total, n_copies, n_per_copy, n_half, colors = _parse_network(network_info)
    G   = nx.from_numpy_array(network_matrix)
    pos = nx.circular_layout(G)

    node_size = max(20,  400 - n_total * 3)
    edge_w    = max(0.5, 3.0 - n_copies / 5)
    font_size = max(6,   10  - n_total // 15)
    fig_dim   = max(10, 8 + n_copies * 0.3)

    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim))
    _draw_network(ax, G, pos, n_total, n_copies, n_per_copy, n_half, colors,
                  node_size, edge_w, font_size)

    ax.legend(handles=_network_legend(n_copies, colors),
              loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)
    ax.set_title(
        f"QL-bit Network (circular) — {network_info['num_ql_bits']} QL-bits  |  "
        f"{n_copies} copies  |  {n_total} total nodes",
        fontsize=13, fontweight='bold',
    )
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path is None:
        save_path = f"ql_network_{network_info['num_ql_bits']}bits_{n_total}nodes_circular.png"
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_copy_colors(n):
    """Return n distinct colors: hardcoded palette for ≤12, colormap beyond that."""
    if n <= len(_COPY_COLORS):
        return list(_COPY_COLORS[:n])
    cmap = plt.get_cmap('hsv')
    return [matplotlib.colors.to_hex(cmap(i / n)) for i in range(n)]


def _parse_network(network_info):
    n_total    = network_info['total_nodes']
    # The base QL-bit (ql_bits[0]) is the unit that gets copied at every level.
    # n_copies = how many of those atoms fill the full network.
    n_per_copy = network_info['ql_bits'][0]['total_nodes']
    n_copies   = n_total // n_per_copy
    n_half     = n_per_copy // 2
    colors     = _make_copy_colors(n_copies)
    return n_total, n_copies, n_per_copy, n_half, colors


def _classify_edges(G, n_copies, n_per_copy, n_half):
    """
    Sort edges into inter-copy, per-copy intra, and intra-coupling lists.
    Node i: copy = i // n_per_copy, position = i % n_per_copy.
    """
    inter    = []
    intra    = {h: [] for h in range(n_copies)}
    coupling = []

    for u, v in G.edges():
        hu, hv = u // n_per_copy, v // n_per_copy
        gu, gv = u % n_per_copy,  v % n_per_copy
        if hu != hv:
            inter.append((u, v))
        elif (gu < n_half) == (gv < n_half):
            intra[hu].append((u, v))
        else:
            coupling.append((u, v))

    return inter, intra, coupling


def _place_cluster(pos, nodes, cx, cy, r):
    n = len(nodes)
    if n == 0:
        return
    if n == 1:
        pos[nodes[0]] = (cx, cy)
        return
    for idx, node in enumerate(nodes):
        a = 2 * np.pi * idx / n
        pos[node] = (cx + r * np.cos(a), cy + r * np.sin(a))


def _network_legend(n_copies, colors):
    shown = min(n_copies, 8)
    items = [Patch(facecolor=colors[h], edgecolor='dimgray', label=f'Copy {h}')
             for h in range(shown)]
    if n_copies > 8:
        items.append(Patch(facecolor='white', edgecolor='dimgray',
                           label=f'… +{n_copies - 8} more copies'))
    items.append(mlines.Line2D([], [], color=_COLOR_INTER, linestyle='--',
                               linewidth=1, label='Inter-copy edges (Cartesian product)'))
    items.append(mlines.Line2D([], [], color=_COLOR_INTRA_C, linewidth=1,
                               label='Intra-QL-bit coupling edges'))
    return items


def _save_and_show(fig, save_path, show_plot):
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
