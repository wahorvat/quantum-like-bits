"""
gates.py

QL gate operations on adjacency matrices.

Based on Section IV of Amati & Scholes, Phys. Rev. A 111, 062203 (2025).

The computational basis is reached via the change-of-basis map

    U_cb = ⊗_q (V_H ⊗ 𝟙_{N_q}),   QL-bit 0 innermost (stride 1)

where V_H = (1/√2)[[1,1],[1,-1]] is the Hadamard matrix.

A single-qubit gate V_g acting on QL-bit q transforms the adjacency matrix as

    R_g = U_g R U_g†

where the graph-space operator is

    U_g^{(q)} = 𝟙_{outer} ⊗ [(V_H^T V_g V_H) ⊗ 𝟙_{N_q}] ⊗ 𝟙_{inner}

with inner = Π_{p<q} (2N_p) and outer = Π_{p>q} (2N_p).

Node-ordering convention
------------------------
QL-bit 0 is the innermost index (stride 1) in the Cartesian-product matrix,
matching the convention in graph_generation.couple_ql_bits.

Public API
----------
compute_basis_change(info)                                  -> np.ndarray
apply_single_qubit_gate(matrix, info, gate_matrix, target)  -> np.ndarray
identity(matrix, info, target_qubit=0)                      -> np.ndarray
pauli_x(matrix, info, target_qubit=0)                       -> np.ndarray
pauli_y(matrix, info, target_qubit=0)                       -> np.ndarray
pauli_z(matrix, info, target_qubit=0)                       -> np.ndarray
hadamard(matrix, info, target_qubit=0)                      -> np.ndarray
t_gate(matrix, info, target_qubit=0)                        -> np.ndarray
phase_gate(matrix, info, phi, target_qubit=0)               -> np.ndarray
apply_gate_sequence(matrix, info, gate_list)                -> np.ndarray
"""

import numpy as np

# -------------------------------------------------------------------
# Standard 2×2 gates in the computational basis
# -------------------------------------------------------------------

_V_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)  # normalised Hadamard

GATES = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    'H': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    'S': np.array([[1, 0], [0, 1j]], dtype=complex),
}


# -------------------------------------------------------------------
# Info-dict helpers
# -------------------------------------------------------------------

def _get_ql_bit_infos(info):
    """
    Return a flat list of per-QL-bit info dicts.

    Accepts either a single-bit dict (from generate_quantum_like_bit) or a
    coupled-network dict (from couple_ql_bits).  In the single-bit case the
    function wraps the dict in a list so callers can iterate uniformly.
    """
    if 'ql_bits' in info:
        return list(info['ql_bits'])
    return [info]


def _get_effective_layout(info):
    """
    Return (infos, eff_sizes, strides) that correctly reflect the current matrix.

    Works for all four info-dict shapes:
      - single-bit dict          (generate_quantum_like_bit)
      - coupled dict             (couple_ql_bits)
      - partially-contracted     (contract_ql_bit)
      - fully-contracted/quotient (minimal_quotient)

    eff_sizes[q] is the number of matrix rows/cols that belong to QL-bit q:
      - 2              if the QL-bit has been contracted (1 node per subgraph)
      - total_nodes    otherwise (2 * N_q)

    strides[q] is read directly from info['current_strides'] when present
    (contract_ql_bit and minimal_quotient both set this field after potentially
    reordering the innermost index).  For uncontracted matrices the strides are
    derived from eff_sizes in the standard way (QL-bit 0 innermost).
    """
    infos = _get_ql_bit_infos(info)
    contracted = info.get('contracted', set())

    eff_sizes = [
        2 if q in contracted else infos[q]['total_nodes']
        for q in range(len(infos))
    ]

    if 'current_strides' in info:
        strides = list(info['current_strides'])
    else:
        strides, s = [], 1
        for sz in eff_sizes:
            strides.append(s)
            s *= sz

    return infos, eff_sizes, strides


# -------------------------------------------------------------------
# Computational basis change  (Eq. 4.1)
# -------------------------------------------------------------------

def compute_basis_change(info):
    """
    Build U_cb = ⊗_q (V_H ⊗ 𝟙_{N_q}) for the full QL system.

    Maps the emergent eigenvector basis to the computational basis:
        U_cb ψ_↑^{(q)} = |0_q⟩ = (a_q, 0)
        U_cb ψ_↓^{(q)} = |1_q⟩ = (0,   a_q)

    The tensor product is built with QL-bit 0 innermost (rightmost factor in
    the Kronecker product), consistent with couple_ql_bits node ordering.

    Parameters
    ----------
    info : dict
        Info dict from generate_quantum_like_bit, couple_ql_bits,
        contract_ql_bit, or minimal_quotient.

    Returns
    -------
    np.ndarray  Complex unitary of shape (n_total, n_total).
    """
    infos, _, _ = _get_effective_layout(info)
    contracted = info.get('contracted', set())

    # Contracted QL-bits have N_q=1 (one representative node per subgraph).
    def _nq(q):
        return 1 if q in contracted else infos[q]['N']

    U = np.kron(_V_H, np.eye(_nq(0)))
    for q in range(1, len(infos)):
        U = np.kron(np.kron(_V_H, np.eye(_nq(q))), U)

    return U.astype(complex)


# -------------------------------------------------------------------
# Core gate application  (Eqs. 4.4, 4.11)
# -------------------------------------------------------------------

def apply_single_qubit_gate(matrix, info, gate_matrix, target_qubit):
    """
    Apply a single-qubit gate to one QL-bit and return the transformed matrix.

    Computes R_g = U_g R U_g† where

        U_g^{(q)} = 𝟙_{outer} ⊗ [(V_H^T V_g V_H) ⊗ 𝟙_{N_q}] ⊗ 𝟙_{inner}

    The conjugation V_H^T V_g V_H converts the gate from the computational
    basis to the emergent-eigenvector basis used by the adjacency matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Adjacency matrix of the QL system (integer or float).
    info : dict
        Info dict from generate_quantum_like_bit, couple_ql_bits,
        contract_ql_bit, or minimal_quotient.
    gate_matrix : array-like, shape (2, 2)
        Unitary gate in the computational basis.
    target_qubit : int
        0-based index of the QL-bit on which to apply the gate.

    Returns
    -------
    np.ndarray  Transformed adjacency matrix (complex dtype).

    Raises
    ------
    ValueError  If target_qubit is out of range.
    """
    infos, eff_sizes, strides = _get_effective_layout(info)
    N_QL = len(infos)

    if not (0 <= target_qubit < N_QL):
        raise ValueError(
            f"target_qubit={target_qubit} out of range for a "
            f"{N_QL}-QL-bit system"
        )

    Vg = np.asarray(gate_matrix, dtype=complex)

    # Contracted QL-bits have one representative node per subgraph (N_q=1),
    # so M_graph = M_local (no kron with identity needed).
    contracted = info.get('contracted', set())
    N_q   = 1 if target_qubit in contracted else infos[target_qubit]['N']
    size_q = eff_sizes[target_qubit]   # 2 if contracted, else 2*N_q

    # Gate conjugated into the emergent basis: V_H^T V_g V_H  (2×2)
    M_local = _V_H.T @ Vg @ _V_H

    # Lift to the local block (size_q × size_q)
    M_graph = np.kron(M_local, np.eye(N_q)).astype(complex)

    # Position the local gate inside the full matrix
    inner = int(strides[target_qubit])
    outer = matrix.shape[0] // (inner * size_q)

    # U_g = 𝟙_outer ⊗ M_graph ⊗ 𝟙_inner
    U_g = np.kron(np.kron(np.eye(outer), M_graph), np.eye(inner)).astype(complex)

    R = matrix.astype(complex)
    return U_g @ R @ U_g.conj().T


# -------------------------------------------------------------------
# Named single-qubit gates
# -------------------------------------------------------------------

def identity(matrix, info, target_qubit=0):
    """Identity gate — returns a complex copy of the matrix unchanged."""
    return apply_single_qubit_gate(matrix, info, GATES['I'], target_qubit)


def pauli_x(matrix, info, target_qubit=0):
    """
    Pauli-X (NOT) gate on QL-bit target_qubit.

    For a single QL-bit this negates the off-diagonal coupling blocks:
        R_X = [[A1, -C], [-C^T, A2]]   (Eq. 4.14)
    """
    return apply_single_qubit_gate(matrix, info, GATES['X'], target_qubit)


def pauli_y(matrix, info, target_qubit=0):
    """Pauli-Y gate on QL-bit target_qubit."""
    return apply_single_qubit_gate(matrix, info, GATES['Y'], target_qubit)


def pauli_z(matrix, info, target_qubit=0):
    """
    Pauli-Z gate on QL-bit target_qubit.

    For a single QL-bit this swaps the two diagonal subgraph blocks:
        R_Z = [[A2, C^T], [C, A1]]
    """
    return apply_single_qubit_gate(matrix, info, GATES['Z'], target_qubit)


def hadamard(matrix, info, target_qubit=0):
    """
    Hadamard gate on QL-bit target_qubit.

    For a single QL-bit:
        R_H = (1/2)[[A1+A2+C+C^T,  A1-A2-C+C^T],
                    [A1-A2+C-C^T,  A1+A2-C-C^T]]
    """
    return apply_single_qubit_gate(matrix, info, GATES['H'], target_qubit)


def t_gate(matrix, info, target_qubit=0):
    """T (π/8) gate on QL-bit target_qubit."""
    return apply_single_qubit_gate(matrix, info, GATES['T'], target_qubit)


def phase_gate(matrix, info, phi, target_qubit=0):
    """
    Phase gate R(φ) = diag(1, e^{iφ}) on QL-bit target_qubit.

    At φ = π/2 this reproduces the S gate; at φ = π/4 the T gate.
    """
    V = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    return apply_single_qubit_gate(matrix, info, V, target_qubit)


# -------------------------------------------------------------------
# Gate sequences / circuits  (Eq. 4.12)
# -------------------------------------------------------------------

def apply_gate_sequence(matrix, info, gate_list):
    """
    Apply a sequence of single-qubit gates as a circuit.

    Gates are applied left-to-right (first gate in the list acts first).
    Each element of gate_list can be:
      - a (gate_matrix, target_qubit) tuple, or
      - a string key from GATES together with a target: ('X', 0), etc.

    Parameters
    ----------
    matrix : np.ndarray
        Initial adjacency matrix.
    info : dict
        Info dict from generate_quantum_like_bit or couple_ql_bits.
    gate_list : list of (gate_or_key, int)
        Sequence of (V_g, target_qubit) pairs.

    Returns
    -------
    np.ndarray  Fully transformed adjacency matrix.

    Examples
    --------
    >>> R_out = apply_gate_sequence(R, info, [('X', 0), ('H', 0)])
    >>> R_out = apply_gate_sequence(R, info, [(GATES['T'], 1), ('Z', 0)])
    """
    R = matrix.astype(complex)
    for spec, target in gate_list:
        Vg = GATES[spec] if isinstance(spec, str) else np.asarray(spec, dtype=complex)
        R = apply_single_qubit_gate(R, info, Vg, target)
    return R
