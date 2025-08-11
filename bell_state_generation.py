
import numpy as np
import matplotlib.pyplot as plt
from k_regular_graph_generator import generate_multiple_k_regular_graphs
from quantumlike_bit_generator import couple_graphs
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
        ql_bit_matrix1 = couple_graphs(k_reg_adj_mats[0], k_reg_adj_mats[1], l)
        ql_bit_matrix2 = couple_graphs(k_reg_adj_mats[2], k_reg_adj_mats[3], l)

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