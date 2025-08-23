"""
Optimized eigenvalue analysis for coupled QL-bits using multiprocessing.
Run this as a standalone Python file for best performance.
"""

import numpy as np
from scipy.sparse.linalg import eigs
from multiprocessing import Pool
import time
import sys

# Import your existing functions (adjust imports as needed)
from quantumlike_bit_generator import *

# Global cache for coupling dictionaries
coupling_cache = {}

def get_cached_coupling(size):
    """Get cached identity coupling dictionary"""
    if size not in coupling_cache:
        coupling_cache[size] = {i: [i] for i in range(size)}
    return coupling_cache[size]

def process_single_sample(args):
    """Process a single sample - designed for multiprocessing"""
    N, k, num_coupling_iterations, sample_id = args
    
    try:
        # Generate ql-bits
        ql_bit1_matrix, _ = generate_ql_bit(N=N, k=k, coupling_spec=1)
        ql_bit2_matrix, _ = generate_ql_bit(N=N, k=k, coupling_spec=1)
        
        # Initial coupling
        ql_bit_coupling = get_cached_coupling(N * 2)
        coupled_matrix = couple_graphs_custom(ql_bit1_matrix, ql_bit2_matrix, ql_bit_coupling)
        
        # Iterative coupling
        for iteration in range(num_coupling_iterations - 1):
            current_size = coupled_matrix.shape[0]
            ql_bit_coupling = get_cached_coupling(current_size)
            coupled_matrix = couple_graphs_custom(coupled_matrix, coupled_matrix, ql_bit_coupling)
        
        # Compute eigenvalues efficiently
        matrix_size = coupled_matrix.shape[0]
        
        if matrix_size > 100:  # Use sparse solver for large matrices
            try:
                eigenvalues, eigenvectors = eigs(coupled_matrix, k=min(20, matrix_size-1), 
                                               which='LM', return_eigenvectors=True)
                
                # Convert to real if nearly real
                if np.allclose(eigenvalues.imag, 0, atol=1e-10):
                    eigenvalues = eigenvalues.real
                    eigenvectors = eigenvectors.real
                
                # Sort by magnitude
                sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
                
            except Exception:
                # Fallback to full decomposition
                eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(coupled_matrix)
                sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
        else:
            # Full decomposition for smaller matrices
            eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(coupled_matrix)
            sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
        
        # Take top 10 eigenvalues and top 5 eigenvectors
        top_eigenvals = eigenvalues[:10]
        top_eigenvecs = eigenvectors[:, :5] if eigenvectors.shape[1] >= 5 else eigenvectors
        
        # Fix sign consistency for eigenvectors
        for vec_idx in range(top_eigenvecs.shape[1]):
            eigenvec = top_eigenvecs[:, vec_idx]
            first_nonzero_idx = np.argmax(np.abs(eigenvec) > 1e-10)
            if eigenvec[first_nonzero_idx] < 0:
                top_eigenvecs[:, vec_idx] *= -1
        
        return sample_id, top_eigenvals, top_eigenvecs, matrix_size
        
    except Exception as e:
        print(f"Error in sample {sample_id}: {e}")
        return sample_id, None, None, None

def run_multiprocess_eigenanalysis(M=50, N=16, k=12, num_coupling_iterations=6, 
                                 num_processes=None, verbose=True):
    """
    Run optimized eigenvalue analysis using multiprocessing.
    
    Args:
        M: Number of samples
        N: Nodes per subgraph
        k: Degree within subgraph  
        num_coupling_iterations: Number of coupling iterations
        num_processes: Number of processes (None = auto)
        verbose: Print progress
    
    Returns:
        tuple: (all_eigenvalues, (avg_top_eigenvals, std_top_eigenvals), avg_top_eigenvecs)
    """
    
    if verbose:
        print(f"Starting multiprocess eigenanalysis...")
        print(f"Parameters: M={M}, N={N}, k={k}, iterations={num_coupling_iterations}")
        print(f"Expected final matrix size: {N * 2 * (2**(num_coupling_iterations-1))}²")
    
    start_time = time.time()
    
    # Pre-populate cache
    if verbose:
        print("Pre-computing coupling dictionaries...")
    for i in range(num_coupling_iterations):
        size = N * 2 * (2 ** i)
        get_cached_coupling(size)
    
    # Prepare arguments for multiprocessing
    args_list = [(N, k, num_coupling_iterations, sample_id) for sample_id in range(M)]
    
    # Run multiprocessing
    if verbose:
        print(f"Starting {M} samples across {num_processes or 'auto'} processes...")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_sample, args_list)
    
    # Process results
    successful_results = [(sample_id, eigenvals, eigenvecs, size) 
                         for sample_id, eigenvals, eigenvecs, size in results 
                         if eigenvals is not None]
    
    if len(successful_results) < M:
        print(f"Warning: {M - len(successful_results)} samples failed")
    
    # Collect data
    all_eigenvalues = []
    all_top_eigenvalues = []
    all_top_eigenvectors = []
    matrix_sizes = []
    
    for sample_id, eigenvals, eigenvecs, matrix_size in successful_results:
        all_eigenvalues.extend(eigenvals)
        all_top_eigenvalues.append(eigenvals[:5])  # Top 5
        all_top_eigenvectors.append(eigenvecs)
        matrix_sizes.append(matrix_size)
    
    # Convert to numpy arrays
    all_eigenvalues = np.array(all_eigenvalues)
    all_top_eigenvalues = np.array(all_top_eigenvalues)
    all_top_eigenvectors = np.array(all_top_eigenvectors)
    
    # Compute statistics
    avg_top_eigenvalues = np.mean(all_top_eigenvalues, axis=0)
    std_top_eigenvalues = np.std(all_top_eigenvalues, axis=0)
    avg_top_eigenvectors = np.mean(all_top_eigenvectors, axis=0)
    
    end_time = time.time()
    
    if verbose:
        print(f"\nAnalysis complete!")
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Successful samples: {len(successful_results)}/{M}")
        print(f"Final matrix size: {matrix_sizes[0] if matrix_sizes else 'N/A'}")
        print(f"Total eigenvalues collected: {len(all_eigenvalues):,}")
        print(f"Speed: {len(all_eigenvalues)/(end_time - start_time):.1f} eigenvalues/second")
    
    return all_eigenvalues, (avg_top_eigenvalues, std_top_eigenvalues), avg_top_eigenvectors

def main():
    """Example usage"""
    
    # Small test
    print("=== SMALL TEST ===")
    all_eigs, (avg_eigs, std_eigs), avg_vecs = run_multiprocess_eigenanalysis(
        M=10, N=4, k=2, num_coupling_iterations=3, verbose=True
    )
    
    print(f"\nTop 5 eigenvalues (mean ± std):")
    for i, (mean_val, std_val) in enumerate(zip(avg_eigs, std_eigs)):
        print(f"  {i+1}: {mean_val:8.4f} ± {std_val:6.4f}")
    
    # Large test (commented out - uncomment to run)
    """
    print("\\n=== LARGE TEST ===")
    all_eigs, (avg_eigs, std_eigs), avg_vecs = run_multiprocess_eigenanalysis(
        M=50, N=16, k=12, num_coupling_iterations=6, verbose=True
    )
    
    print(f"\\nTop 5 eigenvalues (mean ± std):")
    for i, (mean_val, std_val) in enumerate(zip(avg_eigs, std_eigs)):
        print(f"  {i+1}: {mean_val:8.4f} ± {std_val:6.4f}")
    """

if __name__ == "__main__":
    main()