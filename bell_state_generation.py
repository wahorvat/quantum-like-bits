"""
bell_state_generation.py

A module for generating samples of Bell states from QL-bits and 
analyzing eigenvalues and eigenvectors of the Bell states.
"""

import numpy as np
import matplotlib.pyplot as plt
from k_regular_graph_generator import generate_multiple_k_regular_graphs
from quantumlike_bit_generator import couple_graphs_random
from eigenvalue_analysis import compute_eigenvalues_eigenvectors



def sample_eigenvalue_distributions(N, k, l, M=100):
    """
    Sample M instances and collect eigenvalue distributions for k-regular graphs and QL-bits.
    
    Args:
        N (int): Nodes per subgraph
        k (int): Degree within subgraph
        l (int): Coupling degree
        M (int): Number of samples
    
    Returns:
        tuple: (all_k_reg_eigenvalues1, all_k_reg_eigenvalues2, all_ql_bit_eigenvalues1)
    """
    print(f"Sampling {M} instances with N={N}, k={k}, l={l}")
    
    # Collect all eigenvalues
    all_k_reg_eigenvalues1 = []
    all_k_reg_eigenvalues2 = []
    all_ql_bit_eigenvalues1 = []
    all_ql_bit_eigenvalues2 = []
    all_bell_eigenvalues = []
    
    for i in range(M):
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{M} samples...")
        
        # Generate k-regular graphs and QL-bit
        k_reg_adj_mats = generate_multiple_k_regular_graphs(N, k, 4)
        ql_bit_matrix1 = couple_graphs_random(k_reg_adj_mats[0], k_reg_adj_mats[1], l)
        ql_bit_matrix2 = couple_graphs_random(k_reg_adj_mats[2], k_reg_adj_mats[3], l)

        identity_matrix = np.eye(2*N, dtype=int)
        bell_state = np.kron(ql_bit_matrix1, identity_matrix) + np.kron(identity_matrix, ql_bit_matrix2)
        
        # Compute eigenvalues
        k_reg_eigenvalues1, _ = compute_eigenvalues_eigenvectors(k_reg_adj_mats[0])
        k_reg_eigenvalues2, _ = compute_eigenvalues_eigenvectors(k_reg_adj_mats[1])
        ql_bit_eigenvalues1, _ = compute_eigenvalues_eigenvectors(ql_bit_matrix1)
        ql_bit_eigenvalues2, _ = compute_eigenvalues_eigenvectors(ql_bit_matrix2)
        bell_eigenvalues, _ = compute_eigenvalues_eigenvectors(bell_state)
        
        # Collect eigenvalues
        all_k_reg_eigenvalues1.extend(k_reg_eigenvalues1)
        all_k_reg_eigenvalues2.extend(k_reg_eigenvalues2)
        all_ql_bit_eigenvalues1.extend(ql_bit_eigenvalues1)
        all_ql_bit_eigenvalues2.extend(ql_bit_eigenvalues2)
        all_bell_eigenvalues.extend(bell_eigenvalues)
    
    # Convert to numpy arrays
    all_k_reg_eigenvalues1 = np.array(all_k_reg_eigenvalues1)
    all_k_reg_eigenvalues2 = np.array(all_k_reg_eigenvalues2)
    all_ql_bit_eigenvalues1 = np.array(all_ql_bit_eigenvalues1)
    all_ql_bit_eigenvalues2 = np.array(all_ql_bit_eigenvalues2)
    all_bell_eigenvalues = np.array(all_bell_eigenvalues)
    
    print(f"\nCollected eigenvalues:")
    
    # Plot three separate distributions in one window
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5 = axes[1][:2]
    axes[1][2].axis('off') 
    
    # Plot 1: k-regular graph eigenvalues
    ax1.hist(all_k_reg_eigenvalues1, bins=50, alpha=0.7, color='lightblue', 
            edgecolor='black', density=True)
    ax1.set_title(f'k-Regular Graph Eigenvalues\n({len(all_k_reg_eigenvalues1):,} eigenvals from {M} samples)')
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Density')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-3)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics for k-regular
    k_reg_stats = f'Mean: {np.mean(all_k_reg_eigenvalues1):.2f}\nStd: {np.std(all_k_reg_eigenvalues1):.2f}\nMax: {np.max(all_k_reg_eigenvalues1):.2f}\nMin: {np.min(all_k_reg_eigenvalues1):.2f}'
    ax1.text(0.02, 0.98, k_reg_stats, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: k-regular graph eigenvalues
    ax2.hist(all_k_reg_eigenvalues2, bins=50, alpha=0.7, color='lightblue', 
            edgecolor='black', density=True)
    ax2.set_title(f'k-Regular Graph Eigenvalues\n({len(all_k_reg_eigenvalues2):,} eigenvals from {M} samples)')
    ax2.set_xlabel('Eigenvalue')
    ax2.set_ylabel('Density')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-3)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics for k-regular
    k_reg_stats = f'Mean: {np.mean(all_k_reg_eigenvalues2):.2f}\nStd: {np.std(all_k_reg_eigenvalues2):.2f}\nMax: {np.max(all_k_reg_eigenvalues2):.2f}\nMin: {np.min(all_k_reg_eigenvalues2):.2f}'
    ax2.text(0.02, 0.98, k_reg_stats, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    
    # Plot 3: QL-bit1 eigenvalues  
    ax3.hist(all_ql_bit_eigenvalues1, bins=50, alpha=0.7, color='orange', 
            edgecolor='black', density=True)
    ax3.set_title(f'QL-Bit Eigenvalues\n({len(all_ql_bit_eigenvalues1):,} eigenvals from {M} samples)')
    ax3.set_xlabel('Eigenvalue')
    ax3.set_ylabel('Density')
    ax3.set_yscale('log')
    ax3.set_ylim(bottom=1e-3)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics for QL-bits
    ql_bit_stats = f'Mean: {np.mean(all_ql_bit_eigenvalues1):.2f}\nStd: {np.std(all_ql_bit_eigenvalues1):.2f}\nMax: {np.max(all_ql_bit_eigenvalues1):.2f}\nMin: {np.min(all_ql_bit_eigenvalues1):.2f}'
    ax3.text(0.02, 0.98, ql_bit_stats, transform=ax3.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
        # Plot 3: QL-bit eigenvalues  
    ax4.hist(all_ql_bit_eigenvalues2, bins=50, alpha=0.7, color='orange', 
            edgecolor='black', density=True)
    ax4.set_title(f'QL-Bit Eigenvalues\n({len(all_ql_bit_eigenvalues2):,} eigenvals from {M} samples)')
    ax4.set_xlabel('Eigenvalue')
    ax4.set_ylabel('Density')
    ax4.set_yscale('log')
    ax4.set_ylim(bottom=1e-3)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics for QL-bits
    ql_bit_stats = f'Mean: {np.mean(all_ql_bit_eigenvalues2):.2f}\nStd: {np.std(all_ql_bit_eigenvalues2):.2f}\nMax: {np.max(all_ql_bit_eigenvalues2):.2f}\nMin: {np.min(all_ql_bit_eigenvalues2):.2f}'
    ax4.text(0.02, 0.98, ql_bit_stats, transform=ax4.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
        # Plot 3: QL-bit eigenvalues  
    ax5.hist(all_bell_eigenvalues, bins=50, alpha=0.7, color='orange', 
            edgecolor='black', density=True)
    ax5.set_title(f'Bell State Eigenvalues\n({len(all_bell_eigenvalues):,} eigenvals from {M} samples)')
    ax5.set_xlabel('Eigenvalue')
    ax5.set_ylabel('Density')
    ax5.set_yscale('log')
    ax5.set_ylim(bottom=5e-4)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add statistics for QL-bits
    ql_bit_stats = f'Mean: {np.mean(all_bell_eigenvalues):.2f}\nStd: {np.std(all_bell_eigenvalues):.2f}\nMax: {np.max(all_bell_eigenvalues):.2f}\nMin: {np.min(all_bell_eigenvalues):.2f}'
    ax5.text(0.02, 0.98, ql_bit_stats, transform=ax5.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Eigenvalue Analysis: k-Regular vs QL-Bits (N={N}, k={k}, l={l})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return 0

def sample_bell_state_distributions(N, k, l, M=100):
    """
    Sample M bell state instances and analyze eigenvalue/eigenvector distributions.
    
    Args:
        N (int): Nodes per subgraph
        k (int): Degree within subgraph
        l (int): Coupling degree
        M (int): Number of samples
    
    Returns:
        tuple: (all_eigenvalues, top_10_eigenvalues, averaged_eigenvectors)
    """
    print(f"Sampling {M} bell state instances with N={N}, k={k}, l={l}")
    
    # Collect all eigenvalues and top eigenvectors
    all_bell_eigenvalues = []
    all_top_10_eigenvalues = []
    all_top_10_eigenvectors = []
    
    for i in range(M):
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{M} samples...")
        
        # Generate k-regular graphs and QL-bits
        k_reg_adj_mats = generate_multiple_k_regular_graphs(N, k, 4)
        ql_bit_matrix1 = couple_graphs_random(k_reg_adj_mats[0], k_reg_adj_mats[1], l)
        ql_bit_matrix2 = couple_graphs_random(k_reg_adj_mats[2], k_reg_adj_mats[3], l)

        # Create bell state
        identity_matrix = np.eye(2*N, dtype=int)
        bell_state = np.kron(ql_bit_matrix1, identity_matrix) + np.kron(identity_matrix, ql_bit_matrix2)
        
        # Compute eigenvalues and eigenvectors
        bell_eigenvalues, bell_eigenvectors = compute_eigenvalues_eigenvectors(bell_state)
        
        # Sort by eigenvalue magnitude (descending)
        sorted_indices = np.argsort(np.abs(bell_eigenvalues))[::-1]
        sorted_eigenvalues = bell_eigenvalues[sorted_indices]
        sorted_eigenvectors = bell_eigenvectors[:, sorted_indices]
        
        # Keep top 10
        top_10_eigenvals = sorted_eigenvalues[:10]
        top_10_eigenvecs = sorted_eigenvectors[:, :10]
        
        # Store results
        all_bell_eigenvalues.extend(bell_eigenvalues)
        all_top_10_eigenvalues.append(top_10_eigenvals)
        all_top_10_eigenvectors.append(top_10_eigenvecs)
    
    # Convert to numpy arrays
    all_bell_eigenvalues = np.array(all_bell_eigenvalues)
    all_top_10_eigenvalues = np.array(all_top_10_eigenvalues)  # Shape: (M, 10)
    all_top_10_eigenvectors = np.array(all_top_10_eigenvectors)  # Shape: (M, matrix_size, 10)
    
    # Fix eigenvector sign consistency for averaging
    # Make the first non-zero element positive for each eigenvector
    for sample_idx in range(M):
        for vec_idx in range(10):
            eigenvec = all_top_10_eigenvectors[sample_idx, :, vec_idx]
            # Find first non-zero element
            first_nonzero_idx = np.argmax(np.abs(eigenvec) > 1e-10)
            if eigenvec[first_nonzero_idx] < 0:
                all_top_10_eigenvectors[sample_idx, :, vec_idx] *= -1
    
    # Compute average eigenvalues and eigenvectors
    avg_top_10_eigenvalues = np.mean(all_top_10_eigenvalues, axis=0)
    avg_top_10_eigenvectors = np.mean(all_top_10_eigenvectors, axis=0)
    
    # Compute standard deviations for eigenvalues
    std_top_10_eigenvalues = np.std(all_top_10_eigenvalues, axis=0)
    
    print(f"\\nCollected {len(all_bell_eigenvalues):,} bell state eigenvalues from {M} samples")
    
    # Create comprehensive visualization with proper layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main eigenvalue distribution plot
    ax1.hist(all_bell_eigenvalues, bins=75, alpha=0.7, color='purple', 
            edgecolor='black', density=True)
    ax1.set_title(f'Bell State Eigenvalue Distribution\\n({len(all_bell_eigenvalues):,} eigenvals from {M} samples)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Eigenvalue', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-4)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add overall statistics
    bell_stats = f'All Eigenvalues:\\nMean: {np.mean(all_bell_eigenvalues):.3f}\\nStd: {np.std(all_bell_eigenvalues):.3f}\\nMax: {np.max(all_bell_eigenvalues):.3f}\\nMin: {np.min(all_bell_eigenvalues):.3f}'
    ax1.text(0.02, 0.98, bell_stats, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=9)
    
    # Top 10 eigenvalues plot
    eigenval_positions = np.arange(1, 11)
    ax2.errorbar(eigenval_positions, avg_top_10_eigenvalues, yerr=std_top_10_eigenvalues,
                fmt='o-', color='red', linewidth=2, markersize=6, capsize=4)
    ax2.set_title('Top 10 Eigenvalues (Mean ± Std)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Eigenvalue Rank', fontsize=11)
    ax2.set_ylabel('Eigenvalue', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(eigenval_positions)
    
    # Eigenvector magnitude heatmap
    eigenvec_magnitudes = np.abs(avg_top_10_eigenvectors)
    im = ax3.imshow(eigenvec_magnitudes.T, aspect='auto', cmap='viridis', origin='lower')
    ax3.set_title('Average Eigenvector Components\\n|Component Magnitude|', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('Matrix Row/Node Index', fontsize=11)
    ax3.set_ylabel('Eigenvector Rank', fontsize=11)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([f'#{i+1}' for i in range(10)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('|Component|', fontsize=10)
    
    # Eigenvector variance heatmap
    eigenvec_variances = np.var(all_top_10_eigenvectors, axis=0)
    im2 = ax4.imshow(eigenvec_variances.T, aspect='auto', cmap='plasma', origin='lower')
    ax4.set_title('Eigenvector Component Variance\\n(Across Samples)', 
                 fontsize=12, fontweight='bold')
    ax4.set_xlabel('Matrix Row/Node Index', fontsize=11)
    ax4.set_ylabel('Eigenvector Rank', fontsize=11)
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([f'#{i+1}' for i in range(10)])
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax4, shrink=0.8)
    cbar2.set_label('Variance', fontsize=10)
    
    plt.suptitle(f'Bell State Eigenanalysis: N={N}, k={k}, l={l} ({M} samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\\n" + "="*60)
    print("TOP 10 EIGENVALUES (Mean ± Standard Deviation)")
    print("="*60)
    for i, (mean_val, std_val) in enumerate(zip(avg_top_10_eigenvalues, std_top_10_eigenvalues)):
        print(f"Rank {i+1:2d}: {mean_val:8.4f} ± {std_val:6.4f}")
    
    print("\\n" + "="*60)
    print("EIGENVECTOR ANALYSIS SUMMARY")
    print("="*60)
    print(f"Matrix size: {avg_top_10_eigenvectors.shape[0]} x {avg_top_10_eigenvectors.shape[0]}")
    print(f"Top eigenvectors kept: {avg_top_10_eigenvectors.shape[1]}")
    print(f"Samples averaged: {M}")
    
    # Show most significant components for top 3 eigenvectors
    print("\\nMost significant components (top 3 eigenvectors):")
    for vec_idx in range(min(3, 10)):
        eigenvec = avg_top_10_eigenvectors[:, vec_idx]
        top_indices = np.argsort(np.abs(eigenvec))[-5:][::-1]  # Top 5 components
        print(f"\\nEigenvector #{vec_idx+1} (eigenvalue: {avg_top_10_eigenvalues[vec_idx]:.4f}):")
        print("  Top components: [index: value]")
        for idx in top_indices:
            print(f"    [{idx:3d}: {eigenvec[idx]:7.4f}]")
    
    return all_bell_eigenvalues, (avg_top_10_eigenvalues, std_top_10_eigenvalues), avg_top_10_eigenvectors, all_top_10_eigenvectors


def analyze_eigenvector_stability(eigenvectors_samples, top_k=10):
    """
    Analyze stability of eigenvectors across samples.
    
    Args:
        eigenvectors_samples (numpy.ndarray): Shape (M, matrix_size, top_k)
        top_k (int): Number of top eigenvectors to analyze
    
    Returns:
        dict: Stability metrics
    """
    M, matrix_size, _ = eigenvectors_samples.shape
    
    # Compute pairwise correlations between samples for each eigenvector rank
    correlations = np.zeros((top_k, M, M))
    
    for vec_rank in range(top_k):
        for i in range(M):
            for j in range(i, M):
                vec_i = eigenvectors_samples[i, :, vec_rank]
                vec_j = eigenvectors_samples[j, :, vec_rank]
                # Correlation coefficient
                corr = np.corrcoef(vec_i, vec_j)[0, 1]
                correlations[vec_rank, i, j] = corr
                correlations[vec_rank, j, i] = corr
    
    # Compute stability metrics
    stability_metrics = {}
    for vec_rank in range(top_k):
        # Average correlation (excluding diagonal)
        corr_matrix = correlations[vec_rank]
        upper_tri = np.triu(corr_matrix, k=1)
        avg_correlation = np.mean(upper_tri[upper_tri != 0])
        
        # Minimum correlation
        min_correlation = np.min(upper_tri[upper_tri != 0])
        
        stability_metrics[vec_rank] = {
            'avg_correlation': avg_correlation,
            'min_correlation': min_correlation,
            'correlation_std': np.std(upper_tri[upper_tri != 0])
        }
    
    return stability_metrics