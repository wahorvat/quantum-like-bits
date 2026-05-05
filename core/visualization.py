"""
visualization.py

Plotting utilities for quantum-like bits and QL-bit networks.

By default every function saves a PNG to disk (show_plot=False).
Pass show_plot=True to display interactively instead.
Pass save_path='my_file.png' to choose the output path explicitly;
omit it to use an auto-generated name in the current directory.

Edge-weight coloring
--------------------
All four public functions accept two optional keyword arguments:

    edge_weights     : bool   (default False)
        When True, edges are coloured by their matrix entry value rather than
        by edge type.  Edges whose absolute magnitude |w| falls below
        weight_threshold are not rendered.

    weight_threshold : float  (default 1e-3)
        Absolute edge magnitude below which an edge is not drawn.

    Colormaps (both are colorblind-friendly diverging maps):
        Real part      — 'coolwarm'  (blue − / white 0 / red +)
        Imaginary part — 'PuOr'      (purple − / white 0 / orange +)

    When the matrix has a significant imaginary component
    (max|Im| / max|w| > 1e-10) the figure is split into two side-by-side
    panels, each with its own colorbar.

Public API
----------
visualize_ql_bit(ql_matrix, graph_info, *, show_plot, save_path,
                 edge_weights, weight_threshold)
visualize_ql_network(network_matrix, network_info, *, ...)
visualize_ql_network_ring(network_matrix, network_info, *, ...)
visualize_ql_network_circular(network_matrix, network_info, *, ...)

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
# Color palettes
# ---------------------------------------------------------------------------

_COLOR_A = 'lightcoral'   # subgraph 1 of a single QL-bit
_COLOR_B = 'skyblue'      # subgraph 2 of a single QL-bit

_COPY_COLORS = [
    'lightcoral', '#77B1F6', '#EAAC38', '#4CA889',
    '#EA3368',    '#7A339C', '#78AA61', '#F7CD46',
    '#67ABE0',    '#DCA237', '#C76526', '#8761CD',
]

_COLOR_COUPLING = 'gray'
_COLOR_INTER    = 'gray'
_COLOR_INTRA_C  = 'dimgray'

# Colorblind-friendly diverging colormaps for edge-weight mode
_CMAP_REAL = 'coolwarm'   # blue(−) / white(0) / red(+)
_CMAP_IMAG = 'PuOr'       # purple(−) / white(0) / orange(+)

_SPRING_LAYOUT_LIMIT = 200


# ---------------------------------------------------------------------------
# Single QL-bit
# ---------------------------------------------------------------------------

def visualize_ql_bit(ql_matrix, graph_info, show_plot=False, save_path=None,
                     edge_weights=False, weight_threshold=1e-1):
    """
    Visualise a single quantum-like bit.

    Default (edge_weights=False): subgraph 1 in lightcoral, subgraph 2 in
    skyblue, coupling edges in gray.

    With edge_weights=True: edges are coloured by their matrix weight using
    colorblind-friendly diverging colormaps.  Complex matrices (e.g. after a
    Pauli-Y or T gate) produce a two-panel figure showing real and imaginary
    parts separately.

    Parameters
    ----------
    ql_matrix        : np.ndarray   (2N x 2N) adjacency matrix (may be complex)
    graph_info       : dict         Output of generate_quantum_like_bit
    show_plot        : bool         Display interactively (default False)
    save_path        : str | None   PNG path; auto-generated if None
    edge_weights     : bool         Colour edges by weight (default False)
    weight_threshold : float        Absolute magnitude cutoff (default 1e-3)

    Returns
    -------
    matplotlib.figure.Figure
    """
    N = graph_info['N']
    k = graph_info['k']
    l = graph_info['l']

    a_nodes = list(range(N))
    b_nodes = list(range(N, 2 * N))
    a_set   = set(a_nodes)

    if N <= 6:
        fig_size = (12, 6);  node_size = 600;  font_size = 12;  edge_w = 6;  sep = 2.5
    elif N <= 15:
        fig_size = (15, 8);  node_size = 400;  font_size = 10;  edge_w = 4;  sep = 3.0
    else:
        fig_size = (18, 10); node_size = 300;  font_size = 8;   edge_w = 3;  sep = 3.5
    if 2 * N <= 50:
        edge_w *= 2

    # Graph for layout — always real/float so spring_layout works correctly
    G = _matrix_to_layout_graph(ql_matrix) if edge_weights else nx.from_numpy_array(ql_matrix)

    # Node positions (shared between both drawing modes)
    if N == 1:
        pos = {0: (-sep, 0), 1: (sep, 0)}
    else:
        pos = {}
        for node, (x, y) in nx.spring_layout(G.subgraph(a_nodes), seed=42).items():
            pos[node] = (x - sep, y)
        for node, (x, y) in nx.spring_layout(G.subgraph(b_nodes), seed=43).items():
            pos[node] = (x + sep, y)

    internal_a = [(u, v) for u, v in G.edges() if u in a_set     and v in a_set]
    internal_b = [(u, v) for u, v in G.edges() if u not in a_set and v not in a_set]
    cross       = [(u, v) for u, v in G.edges() if (u in a_set)  != (v in a_set)]

    if not edge_weights:
        # ── Categorical drawing (original behaviour) ─────────────────────────
        coupling_edges = graph_info['coupling_edges']
        fig, ax = plt.subplots(figsize=fig_size)

        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2,
                               edge_color='lightgray', ax=ax)
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
        nx.draw_networkx_nodes(G, pos, nodelist=a_nodes, node_color=_COLOR_A,
                               node_size=node_size, ax=ax, **node_style)
        nx.draw_networkx_nodes(G, pos, nodelist=b_nodes, node_color=_COLOR_B,
                               node_size=node_size, ax=ax, **node_style)

        labels = {i: f'$a_{{{i}}}$' for i in range(N)}
        labels.update({N + i: f'$b_{{{i}}}$' for i in range(N)})
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size,
                                font_color='white', font_weight='bold', ax=ax)

        ax.set_title(
            f"Quantum-Like Bit — {k}-regular, N={N} nodes per subgraph, "
            f"l={l} coupling per node",
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
        ax.legend(handles=legend, loc='upper right',
                  bbox_to_anchor=(1.15, 1), fontsize=10)

    else:
        # ── Edge-weight coloring ──────────────────────────────────────────────
        matrix    = np.asarray(ql_matrix, dtype=complex)
        has_imag  = _has_significant_imag(matrix)
        base_title = (f"Quantum-Like Bit — {k}-regular, N={N}, l={l}"
                      f"  |  edge weights")

        node_legend = [
            Patch(facecolor=_COLOR_A, edgecolor='dimgray',
                  label=f'Subgraph 1  ({N} a-nodes)'),
            Patch(facecolor=_COLOR_B, edgecolor='dimgray',
                  label=f'Subgraph 2  ({N} b-nodes)'),
        ]

        if has_imag:
            fig, (ax_r, ax_i) = plt.subplots(
                1, 2, figsize=(fig_size[0] * 2.0, fig_size[1]))
            panels = [
                (ax_r, 'real', _CMAP_REAL, 'Real part'),
                (ax_i, 'imag', _CMAP_IMAG, 'Imaginary part'),
            ]
            fig.suptitle(base_title, fontsize=14, fontweight='bold', y=1.01)
        else:
            fig, ax_r = plt.subplots(
                figsize=(fig_size[0] * 1.2, fig_size[1]))
            panels = [(ax_r, 'real', _CMAP_REAL, 'Edge weight')]
            ax_r.set_title(base_title, fontsize=14, fontweight='bold', pad=20)

        for ax, component, cmap_name, panel_title in panels:
            _draw_ql_bit_weighted_panel(
                ax, fig, matrix, component, N, G, pos,
                internal_a, internal_b, cross,
                node_size, edge_w, font_size,
                weight_threshold, cmap_name, panel_title,
            )
            ax.legend(handles=node_legend, loc='upper left',
                      fontsize=9, framealpha=0.7)

    plt.tight_layout()
    if save_path is None:
        suffix = '_weighted' if edge_weights else ''
        save_path = f'ql_bit_N{N}_k{k}_l{l}{suffix}.png'
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# QL-bit networks — shared drawing cores
# ---------------------------------------------------------------------------

def _draw_network(ax, G, pos, n_total, n_copies, n_per_copy, n_half, colors,
                  node_size, edge_w, font_size):
    """Categorical edge drawing (original behaviour)."""
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

    _draw_network_nodes_and_labels(ax, G, pos, n_total, n_copies, n_per_copy,
                                   colors, node_size, font_size)


# ---------------------------------------------------------------------------
# QL-bit networks (spring layout)
# ---------------------------------------------------------------------------

def visualize_ql_network(network_matrix, network_info, show_plot=False, save_path=None,
                         edge_weights=False, weight_threshold=1e-1):
    """
    Spring-layout visualisation of a coupled QL-bit network.

    Automatically falls back to circular layout when the network has more
    than 200 nodes, where spring layout becomes prohibitively slow.

    Parameters
    ----------
    network_matrix   : np.ndarray   Combined adjacency matrix from couple_ql_bits
    network_info     : dict         Output dict from couple_ql_bits
    show_plot        : bool         Display interactively (default False)
    save_path        : str | None   PNG path; auto-generated if None
    edge_weights     : bool         Colour edges by weight (default False)
    weight_threshold : float        Absolute magnitude cutoff (default 1e-3)

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_total, n_copies, n_per_copy, n_half, colors = _parse_network(network_info)
    G = (_matrix_to_layout_graph(network_matrix) if edge_weights
         else nx.from_numpy_array(network_matrix))

    if n_total <= _SPRING_LAYOUT_LIMIT:
        main_r    = max(2.0, n_copies * 0.6)
        cluster_r = min(0.5, main_r / (n_copies + 1))
        init_pos  = {}
        for i in range(n_total):
            h, g   = i // n_per_copy, i % n_per_copy
            angle  = 2 * np.pi * h / n_copies
            cx, cy = main_r * np.cos(angle), main_r * np.sin(angle)
            inner  = 2 * np.pi * g / n_per_copy
            init_pos[i] = (cx + cluster_r * np.cos(inner),
                           cy + cluster_r * np.sin(inner))
        iterations = max(30, 80 - n_total // 5)
        pos = nx.spring_layout(G, pos=init_pos, k=0.3,
                               iterations=iterations, seed=42)
    else:
        pos = nx.circular_layout(G)

    node_size = max(30,  400 - n_total * 3)
    edge_w    = max(0.5, 3.0 - n_copies / 5)
    if n_total <= 50:
        edge_w *= 2
    font_size = max(6,   10  - n_total // 15)

    base_title = (
        f"QL-bit Network — {network_info['num_ql_bits']} QL-bits  |  "
        f"{n_copies} copies  |  {n_total} total nodes"
        + ('' if n_total <= _SPRING_LAYOUT_LIMIT else '  [circular fallback]')
    )

    fig = _make_network_figure(
        G, pos, network_matrix, network_info, n_total, n_copies, n_per_copy,
        n_half, colors, node_size, edge_w, font_size,
        edge_weights, weight_threshold, base_title, figsize=(14, 10),
    )

    if save_path is None:
        suffix = '_weighted' if edge_weights else '_spring'
        save_path = (f"ql_network_{network_info['num_ql_bits']}bits"
                     f"_{n_total}nodes{suffix}.png")
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# QL-bit networks (ring layout)
# ---------------------------------------------------------------------------

def visualize_ql_network_ring(network_matrix, network_info, show_plot=False,
                               save_path=None, edge_weights=False,
                               weight_threshold=1e-1):
    """
    Ring-layout visualisation of a coupled QL-bit network.

    Each copy occupies an equal arc; nodes within each arc are arranged
    in a small circle.

    Parameters
    ----------
    network_matrix   : np.ndarray   Combined adjacency matrix from couple_ql_bits
    network_info     : dict         Output dict from couple_ql_bits
    show_plot        : bool         Display interactively (default False)
    save_path        : str | None   PNG path; auto-generated if None
    edge_weights     : bool         Colour edges by weight (default False)
    weight_threshold : float        Absolute magnitude cutoff (default 1e-3)

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_total, n_copies, n_per_copy, n_half, colors = _parse_network(network_info)
    G = (_matrix_to_layout_graph(network_matrix) if edge_weights
         else nx.from_numpy_array(network_matrix))

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
    if n_total <= 50:
        edge_w *= 2
    font_size = max(6,   10  - n_total // 20)
    fig_dim   = max(10,  main_r * 2 + 3)

    base_title = (
        f"QL-bit Network Ring — {network_info['num_ql_bits']} QL-bits  |  "
        f"{n_copies} copies  |  {n_total} total nodes"
    )

    fig = _make_network_figure(
        G, pos, network_matrix, network_info, n_total, n_copies, n_per_copy,
        n_half, colors, node_size, edge_w, font_size,
        edge_weights, weight_threshold, base_title,
        figsize=(fig_dim, fig_dim), aspect_equal=True,
    )

    # Annotate copy centres
    for ax in fig.axes:
        if not _is_colorbar_ax(ax):
            for h in range(n_copies):
                angle  = 2 * np.pi * h / n_copies - np.pi / 2
                cx, cy = main_r * np.cos(angle), main_r * np.sin(angle)
                ax.text(cx, cy, f'C{h}', fontsize=9, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.6))

    if save_path is None:
        suffix = '_weighted' if edge_weights else '_ring'
        save_path = (f"ql_network_{network_info['num_ql_bits']}bits"
                     f"_{n_total}nodes{suffix}.png")
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# QL-bit networks (NetworkX circular layout)
# ---------------------------------------------------------------------------

def visualize_ql_network_circular(network_matrix, network_info, show_plot=False,
                                   save_path=None, edge_weights=False,
                                   weight_threshold=1e-1):
    """
    Circular-layout visualisation using nx.circular_layout.

    All nodes placed on a single circle in index order.
    Copy membership shown through node colour.

    Parameters
    ----------
    network_matrix   : np.ndarray   Combined adjacency matrix from couple_ql_bits
    network_info     : dict         Output dict from couple_ql_bits
    show_plot        : bool         Display interactively (default False)
    save_path        : str | None   PNG path; auto-generated if None
    edge_weights     : bool         Colour edges by weight (default False)
    weight_threshold : float        Absolute magnitude cutoff (default 1e-3)

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_total, n_copies, n_per_copy, n_half, colors = _parse_network(network_info)
    G = (_matrix_to_layout_graph(network_matrix) if edge_weights
         else nx.from_numpy_array(network_matrix))
    pos = nx.circular_layout(G)

    node_size = max(20,  400 - n_total * 3)
    edge_w    = max(0.5, 3.0 - n_copies / 5)
    if n_total <= 50:
        edge_w *= 2
    font_size = max(6,   10  - n_total // 15)
    fig_dim   = max(10,  8 + n_copies * 0.3)

    base_title = (
        f"QL-bit Network (circular) — {network_info['num_ql_bits']} QL-bits  |  "
        f"{n_copies} copies  |  {n_total} total nodes"
    )

    fig = _make_network_figure(
        G, pos, network_matrix, network_info, n_total, n_copies, n_per_copy,
        n_half, colors, node_size, edge_w, font_size,
        edge_weights, weight_threshold, base_title,
        figsize=(fig_dim, fig_dim), aspect_equal=True,
    )

    if save_path is None:
        suffix = '_weighted' if edge_weights else '_circular'
        save_path = (f"ql_network_{network_info['num_ql_bits']}bits"
                     f"_{n_total}nodes{suffix}.png")
    _save_and_show(fig, save_path, show_plot)
    return fig


# ---------------------------------------------------------------------------
# Network figure factory (shared by all three network functions)
# ---------------------------------------------------------------------------

def _make_network_figure(G, pos, matrix, network_info, n_total, n_copies,
                          n_per_copy, n_half, colors, node_size, edge_w,
                          font_size, edge_weights, weight_threshold,
                          base_title, figsize, aspect_equal=False):
    """Create the complete network figure (categorical or weighted)."""
    if not edge_weights:
        fig, ax = plt.subplots(figsize=figsize)
        _draw_network(ax, G, pos, n_total, n_copies, n_per_copy, n_half,
                      colors, node_size, edge_w, font_size)
        ax.legend(handles=_network_legend(n_copies, colors),
                  loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)
        ax.set_title(base_title, fontsize=13, fontweight='bold')
        ax.axis('off')
        if aspect_equal:
            ax.set_aspect('equal')
        plt.tight_layout()
        return fig

    # ── Weighted mode ────────────────────────────────────────────────────────
    wt_matrix = np.asarray(matrix, dtype=complex)
    has_imag  = _has_significant_imag(wt_matrix)

    if has_imag:
        fig, (ax_r, ax_i) = plt.subplots(
            1, 2, figsize=(figsize[0] * 2.0, figsize[1]))
        panels = [
            (ax_r, 'real', _CMAP_REAL, 'Real part'),
            (ax_i, 'imag', _CMAP_IMAG, 'Imaginary part'),
        ]
        fig.suptitle(base_title, fontsize=13, fontweight='bold', y=1.01)
    else:
        fig, ax_r = plt.subplots(figsize=(figsize[0] * 1.2, figsize[1]))
        panels = [(ax_r, 'real', _CMAP_REAL, 'Edge weight')]
        ax_r.set_title(base_title, fontsize=13, fontweight='bold')

    copy_legend = _network_legend_nodes_only(n_copies, colors)

    for ax, component, cmap_name, panel_title in panels:
        _draw_network_weighted_panel(
            ax, fig, G, pos, wt_matrix, component,
            n_total, n_copies, n_per_copy, n_half,
            colors, node_size, edge_w, font_size,
            weight_threshold, cmap_name, panel_title,
        )
        ax.legend(handles=copy_legend, loc='upper right',
                  bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.7)
        ax.axis('off')
        if aspect_equal:
            ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Weighted panel drawing helpers
# ---------------------------------------------------------------------------

def _draw_ql_bit_weighted_panel(ax, fig, matrix, component, N, G, pos,
                                 internal_a, internal_b, cross,
                                 node_size, edge_w, font_size,
                                 threshold, cmap_name, panel_title):
    """Draw one weight-coloured panel for a single QL-bit."""
    a_nodes = list(range(N))
    b_nodes = list(range(N, 2 * N))
    cmap    = matplotlib.cm.get_cmap(cmap_name)

    # Partition all edge categories by threshold (uses full complex magnitude)
    vis_a, hid_a = _partition_edges(matrix, internal_a, threshold)
    vis_b, hid_b = _partition_edges(matrix, internal_b, threshold)
    vis_c, hid_c = _partition_edges(matrix, cross,       threshold)

    # Global symmetric bounds from visible edges
    all_vis = vis_a + vis_b + vis_c
    vmin, vmax = _weight_vbounds(matrix, all_vis, component)

    # Above-threshold edges: coloured by weight, width encodes edge category
    for edges, width in ((vis_a, edge_w), (vis_b, edge_w), (vis_c, 2.0)):
        _draw_edges_weighted(ax, G, pos, matrix, edges, component,
                             cmap, vmin, vmax, width, 0.85)

    # Nodes
    node_style = {'edgecolors': 'dimgray', 'alpha': 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=a_nodes, node_color=_COLOR_A,
                           node_size=node_size, ax=ax, **node_style)
    nx.draw_networkx_nodes(G, pos, nodelist=b_nodes, node_color=_COLOR_B,
                           node_size=node_size, ax=ax, **node_style)

    labels = {i: f'$a_{{{i}}}$' for i in range(N)}
    labels.update({N + i: f'$b_{{{i}}}$' for i in range(N)})
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size,
                            font_color='white', font_weight='bold', ax=ax)

    ax.set_title(panel_title, fontsize=12, pad=8)
    ax.axis('off')
    _add_weight_colorbar(fig, ax, cmap, vmin, vmax, panel_title)


def _draw_network_weighted_panel(ax, fig, G, pos, matrix, component,
                                  n_total, n_copies, n_per_copy, n_half,
                                  colors, node_size, edge_w, font_size,
                                  threshold, cmap_name, panel_title):
    """Draw one weight-coloured panel for a QL-bit network."""
    cmap = matplotlib.cm.get_cmap(cmap_name)
    inter, intra, coupling = _classify_edges(G, n_copies, n_per_copy, n_half)

    all_edges = (inter
                 + [e for h in range(n_copies) for e in intra[h]]
                 + coupling)

    vis_all, hid_all = _partition_edges(matrix, all_edges, threshold)
    vmin, vmax = _weight_vbounds(matrix, vis_all, component)

    # Above-threshold: coloured, dashed style preserved for inter-copy edges
    vis_set = set(vis_all)
    vis_inter    = [e for e in inter    if e in vis_set]
    vis_coupling = [e for e in coupling if e in vis_set]

    if vis_inter:
        _draw_edges_weighted(ax, G, pos, matrix, vis_inter, component,
                             cmap, vmin, vmax, 0.8, 0.4,
                             extra_kwargs={'style': 'dashed'})
    if vis_coupling:
        _draw_edges_weighted(ax, G, pos, matrix, vis_coupling, component,
                             cmap, vmin, vmax, 1.0, 0.7)

    for h in range(n_copies):
        vis_intra_h = [e for e in intra[h] if e in vis_set]
        if vis_intra_h:
            _draw_edges_weighted(ax, G, pos, matrix, vis_intra_h, component,
                                 cmap, vmin, vmax, edge_w, 0.8)

    _draw_network_nodes_and_labels(ax, G, pos, n_total, n_copies, n_per_copy,
                                   colors, node_size, font_size)
    ax.set_title(panel_title, fontsize=12, pad=8)
    _add_weight_colorbar(fig, ax, cmap, vmin, vmax, panel_title)


# ---------------------------------------------------------------------------
# Shared node / label drawing
# ---------------------------------------------------------------------------

def _draw_network_nodes_and_labels(ax, G, pos, n_total, n_copies, n_per_copy,
                                    colors, node_size, font_size):
    for h in range(n_copies):
        nodes = list(range(h * n_per_copy, (h + 1) * n_per_copy))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors[h],
                               node_size=node_size, edgecolors='dimgray',
                               alpha=0.9, ax=ax)
    if n_total <= 36:
        nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(n_total)},
                                font_size=font_size, font_color='white',
                                font_weight='bold', ax=ax)


# ---------------------------------------------------------------------------
# Edge-weight helpers
# ---------------------------------------------------------------------------

def _has_significant_imag(matrix, tol=1e-10):
    """True when max|Im(matrix)| / max|matrix| exceeds tol."""
    if not np.iscomplexobj(matrix):
        return False
    max_abs = np.max(np.abs(matrix))
    if max_abs < 1e-15:
        return False
    return np.max(np.abs(matrix.imag)) / max_abs > tol


def _matrix_to_layout_graph(matrix, tiny=1e-10):
    """Binary graph for layout: edge exists where |w|/max|w| > tiny."""
    mag = np.abs(np.asarray(matrix, dtype=complex))
    max_m = mag.max()
    if max_m < 1e-15:
        return nx.from_numpy_array(np.zeros(mag.shape, dtype=float))
    return nx.from_numpy_array((mag / max_m > tiny).astype(float))


def _partition_edges(matrix, edges, threshold):
    """Split edges into (visible, hidden) by absolute |w| >= threshold."""
    if not edges:
        return [], []
    mags = np.array([abs(matrix[u, v]) for u, v in edges])
    keep = mags >= threshold
    return ([e for e, k in zip(edges, keep) if     k],
            [e for e, k in zip(edges, keep) if not k])


def _weight_vbounds(matrix, edges, component):
    """Symmetric (vmin, vmax) centred at 0 from the given component."""
    if not edges:
        return -1.0, 1.0
    vals    = np.array([getattr(complex(matrix[u, v]), component)
                        for u, v in edges])
    abs_max = max(np.max(np.abs(vals)), 1e-15)
    return -abs_max, abs_max


def _draw_edges_weighted(ax, G, pos, matrix, edges, component,
                          cmap, vmin, vmax, width, alpha,
                          extra_kwargs=None):
    """Draw edges coloured by scalar component using a colormap."""
    if not edges:
        return
    scalars = [getattr(complex(matrix[u, v]), component) for u, v in edges]
    kwargs  = dict(
        edgelist=edges,
        edge_color=scalars,
        edge_cmap=cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
        width=width,
        alpha=alpha,
        ax=ax,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    nx.draw_networkx_edges(G, pos, **kwargs)


def _add_weight_colorbar(fig, ax, cmap, vmin, vmax, label):
    """Add a labelled colorbar to the right of ax."""
    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap,
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02, fraction=0.04)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Shared helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _make_copy_colors(n):
    """Return n distinct colors: hardcoded palette for ≤12, colormap beyond that."""
    if n <= len(_COPY_COLORS):
        return list(_COPY_COLORS[:n])
    cmap = plt.get_cmap('hsv')
    return [matplotlib.colors.to_hex(cmap(i / n)) for i in range(n)]


def _parse_network(network_info):
    n_total = network_info['total_nodes']

    if 'current_strides' in network_info:
        # After contraction the innermost QL-bit (stride 1) may have shrunk to
        # 2 representative nodes.  Find it and use its effective size.
        strides    = network_info['current_strides']
        contracted = network_info.get('contracted', set())
        infos      = network_info['ql_bits']
        q0         = strides.index(1)
        n_per_copy = 2 if q0 in contracted else infos[q0]['total_nodes']
    else:
        n_per_copy = network_info['ql_bits'][0]['total_nodes']

    n_copies = n_total // n_per_copy
    n_half   = n_per_copy // 2
    colors   = _make_copy_colors(n_copies)
    return n_total, n_copies, n_per_copy, n_half, colors


def _classify_edges(G, n_copies, n_per_copy, n_half):
    """Sort edges into inter-copy, per-copy intra, and intra-coupling lists."""
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
                               linewidth=1,
                               label='Inter-copy edges (Cartesian product)'))
    items.append(mlines.Line2D([], [], color=_COLOR_INTRA_C, linewidth=1,
                               label='Intra-QL-bit coupling edges'))
    return items


def _network_legend_nodes_only(n_copies, colors):
    """Compact node-colour legend for the weighted mode (no edge entries)."""
    shown = min(n_copies, 8)
    items = [Patch(facecolor=colors[h], edgecolor='dimgray', label=f'Copy {h}')
             for h in range(shown)]
    if n_copies > 8:
        items.append(Patch(facecolor='white', edgecolor='dimgray',
                           label=f'… +{n_copies - 8} more copies'))
    return items


def _is_colorbar_ax(ax):
    """Heuristic: colorbar axes are very narrow relative to their figure."""
    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    bbox = ax.get_position()
    return bbox.width / fig_w < 0.05


def _save_and_show(fig, save_path, show_plot):
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
