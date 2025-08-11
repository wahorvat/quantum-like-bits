"""
eigenvalue_analysis.py

A module for analyzing eigenvalues and eigenvectors of coupled 
k-regular graphs (ql-bits and the Cartesian product of al-bit graphs).
Includes visualization of eigenvalue distributions.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_eigenvalues_eigenvectors(adjacency_matrix):
    """
    Compute eigenvalues and eigenvectors of the adjacency matrix.
    
    Args:
        adjacency_matrix (numpy.ndarray): Adjacency matrix of the graph
    
    Returns:
        tuple: (eigenvalues, eigenvectors)
            - eigenvalues (numpy.ndarray): Sorted eigenvalues (descending order)
            - eigenvectors (numpy.ndarray): Corresponding eigenvectors (columns)
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors



def visualize_eigenvalue_distribution(coupled_matrix, graph_info, show_plot=True, save_path=None):
    """
    Visualize the eigenvalue distributions of the two subgraphs and the coupled graph.
    
    Args:
        coupled_matrix (numpy.ndarray): Adjacency matrix of the coupled graph
        graph_info (dict): Information about the graph
        show_plot (bool): Whether to display the plot
        save_path (str): Path to save the plot (optional)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    N = graph_info['subgraph_size']
    k = graph_info['subgraph_degree']
    m = graph_info['coupling_degree']
    
    # Extract subgraph adjacency matrices
    subgraph1_matrix = coupled_matrix[:N, :N]
    subgraph2_matrix = coupled_matrix[N:, N:]
    
    # Compute eigenvalues for each matrix
    eigenvals_subgraph1, _ = compute_eigenvalues_eigenvectors(subgraph1_matrix)
    eigenvals_subgraph2, _ = compute_eigenvalues_eigenvectors(subgraph2_matrix)
    eigenvals_coupled, _ = compute_eigenvalues_eigenvectors(coupled_matrix)
    
    # Create three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Subgraph 1 eigenvalues
    axes[0].hist(eigenvals_subgraph1, bins=min(15, len(eigenvals_subgraph1)//2 + 1), 
                alpha=0.7, color='lightblue', edgecolor='black')
    axes[0].set_title(f'Subgraph 1 Eigenvalues\n({k}-Regular, {N} nodes)')
    axes[0].set_xlabel('Eigenvalue')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics text
    stats1_text = f'Max: {eigenvals_subgraph1[0]:.2f}\nMin: {eigenvals_subgraph1[-1]:.2f}\nRange: {eigenvals_subgraph1[0] - eigenvals_subgraph1[-1]:.2f}'
    axes[0].text(0.02, 0.98, stats1_text, transform=axes[0].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Subgraph 2 eigenvalues
    axes[1].hist(eigenvals_subgraph2, bins=min(15, len(eigenvals_subgraph2)//2 + 1), 
                alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title(f'Subgraph 2 Eigenvalues\n({k}-Regular, {N} nodes)')
    axes[1].set_xlabel('Eigenvalue')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics text
    stats2_text = f'Max: {eigenvals_subgraph2[0]:.2f}\nMin: {eigenvals_subgraph2[-1]:.2f}\nRange: {eigenvals_subgraph2[0] - eigenvals_subgraph2[-1]:.2f}'
    axes[1].text(0.02, 0.98, stats2_text, transform=axes[1].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Coupled graph eigenvalues
    axes[2].hist(eigenvals_coupled, bins=min(20, len(eigenvals_coupled)//2 + 1), 
                alpha=0.7, color='gold', edgecolor='black')
    axes[2].set_title(f'Coupled Graph Eigenvalues\n(Degree {k+m}, {2*N} nodes)')
    axes[2].set_xlabel('Eigenvalue')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics text
    stats_coupled_text = f'Max: {eigenvals_coupled[0]:.2f}\nMin: {eigenvals_coupled[-1]:.2f}\nRange: {eigenvals_coupled[0] - eigenvals_coupled[-1]:.2f}\nGap: {eigenvals_coupled[0] - eigenvals_coupled[1]:.2f}'
    axes[2].text(0.02, 0.98, stats_coupled_text, transform=axes[2].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title
    fig.suptitle(f'Eigenvalue Distributions: {k}-Regular Subgraphs (N={N}) with Coupling m={m}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig, (eigenvals_subgraph1, eigenvals_subgraph2, eigenvals_coupled)

