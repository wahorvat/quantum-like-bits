"""
tests/test_gates.py  —  fast smoke tests for core/gates.py

Kept deliberately small (N=6 single-bit, one 2-bit product) so the full
suite runs in a few seconds.  Each test targets one structural guarantee:

  BasisChange   — shape, unitarity, eigenvector mapping
  PauliX        — block formula (Eq. 4.14) + self-inverse + eigenspectrum
  PauliZ        — block formula (swap) + self-inverse + eigenspectrum
  Hadamard      — block formula (Eq. 4.15 × 1/2) + self-inverse + spectrum
  PauliY/T/Phase— eigenspectrum preservation only (output is complex)
  TwoBit        — gate on each qubit of a 6×8-node product (spectrum only)
  Circuits      — H X H = Z,  H Z H = X,  multi-qubit cancellation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from core.graph_generation import generate_quantum_like_bit, couple_ql_bits
from core.gates import (
    GATES, compute_basis_change, apply_single_qubit_gate, apply_gate_sequence,
    identity, pauli_x, pauli_y, pauli_z, hadamard, t_gate, phase_gate,
)

# ---------------------------------------------------------------------------
# Shared fixtures — built once, reused across all tests
# ---------------------------------------------------------------------------

N = 6           # nodes per subgraph (small but valid: N*k even, k < N)
K, L = 4, 2    # k=4-regular, l=2 coupling

@pytest.fixture(scope="module")
def single():
    return generate_quantum_like_bit(N, K, L)   # (R, info)

@pytest.fixture(scope="module")
def two_bit():
    a = generate_quantum_like_bit(N, K, L)
    b = generate_quantum_like_bit(N, K, L)
    return couple_ql_bits(a, b)                  # (R, info)

def eigs(M):
    return np.sort(np.linalg.eigvalsh(M).real)

def blocks(R):
    """Return A1, C, A2 for a (2N × 2N) single-bit matrix."""
    return R[:N, :N], R[:N, N:], R[N:, N:]


# ---------------------------------------------------------------------------
# compute_basis_change
# ---------------------------------------------------------------------------

class TestBasisChange:

    def test_shape(self, single):
        R, info = single
        assert compute_basis_change(info).shape == (2*N, 2*N)

    def test_unitary(self, single):
        R, info = single
        U = compute_basis_change(info)
        assert np.allclose(U @ U.conj().T, np.eye(2*N), atol=1e-12)

    def test_maps_psi_up_to_computational_basis(self, single):
        """U_cb (a1, a1)/√2 = (a1, 0)."""
        R, info = single
        U = compute_basis_change(info)
        a1 = np.linalg.eigh(R[:N, :N].astype(float))[1][:, -1]
        psi_up = np.concatenate([a1,  a1]) / np.sqrt(2)
        psi_dn = np.concatenate([a1, -a1]) / np.sqrt(2)
        cb_up = U @ psi_up
        cb_dn = U @ psi_dn
        assert np.allclose(cb_up[:N], a1,          atol=1e-12)
        assert np.allclose(cb_up[N:], np.zeros(N), atol=1e-12)
        assert np.allclose(cb_dn[:N], np.zeros(N), atol=1e-12)
        assert np.allclose(cb_dn[N:], a1,          atol=1e-12)

    def test_two_bit_unitary(self, two_bit):
        R, info = two_bit
        U = compute_basis_change(info)
        n = U.shape[0]
        assert np.allclose(U @ U.conj().T, np.eye(n), atol=1e-11)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestIdentity:

    def test_single(self, single):
        R, info = single
        assert np.allclose(identity(R, info), R.astype(complex), atol=1e-12)

    def test_two_bit_both_qubits(self, two_bit):
        R, info = two_bit
        for q in (0, 1):
            assert np.allclose(identity(R, info, q), R.astype(complex), atol=1e-12)


# ---------------------------------------------------------------------------
# Pauli-X
# ---------------------------------------------------------------------------

class TestPauliX:

    def test_block_formula(self, single):
        """R_X = [[A1, -C], [-C^T, A2]]"""
        R, info = single
        Rx = pauli_x(R, info)
        A1, C, A2 = blocks(R)
        assert np.allclose(Rx[:N, :N], A1,  atol=1e-12)
        assert np.allclose(Rx[N:, N:], A2,  atol=1e-12)
        assert np.allclose(Rx[:N, N:], -C,  atol=1e-12)
        assert np.allclose(Rx[N:, :N], -C.T, atol=1e-12)

    def test_self_inverse(self, single):
        R, info = single
        assert np.allclose(pauli_x(pauli_x(R, info), info), R.astype(complex), atol=1e-12)

    def test_eigenspectrum(self, single):
        R, info = single
        assert np.allclose(eigs(R), eigs(pauli_x(R, info)), atol=1e-10)

    def test_two_bit_eigenspectrum(self, two_bit):
        R, info = two_bit
        for q in (0, 1):
            assert np.allclose(eigs(R), eigs(pauli_x(R, info, q)), atol=1e-10)

    def test_out_of_range(self, single):
        R, info = single
        with pytest.raises(ValueError):
            pauli_x(R, info, target_qubit=1)


# ---------------------------------------------------------------------------
# Pauli-Z
# ---------------------------------------------------------------------------

class TestPauliZ:

    def test_block_formula(self, single):
        """R_Z = [[A2, C^T], [C, A1]]  (subgraph swap)."""
        R, info = single
        Rz = pauli_z(R, info)
        A1, C, A2 = blocks(R)
        assert np.allclose(Rz[:N, :N], A2,  atol=1e-12)
        assert np.allclose(Rz[N:, N:], A1,  atol=1e-12)
        assert np.allclose(Rz[:N, N:], C.T, atol=1e-12)
        assert np.allclose(Rz[N:, :N], C,   atol=1e-12)

    def test_self_inverse(self, single):
        R, info = single
        assert np.allclose(pauli_z(pauli_z(R, info), info), R.astype(complex), atol=1e-12)

    def test_eigenspectrum(self, single):
        R, info = single
        assert np.allclose(eigs(R), eigs(pauli_z(R, info)), atol=1e-10)

    def test_two_bit_eigenspectrum(self, two_bit):
        R, info = two_bit
        for q in (0, 1):
            assert np.allclose(eigs(R), eigs(pauli_z(R, info, q)), atol=1e-10)


# ---------------------------------------------------------------------------
# Hadamard
# ---------------------------------------------------------------------------

class TestHadamard:

    def test_block_formula(self, single):
        """R_H = (1/2)[[A1+A2+C+C^T, A1-A2-C+C^T], [A1-A2+C-C^T, A1+A2-C-C^T]]."""
        R, info = single
        Rh = hadamard(R, info)
        A1 = R[:N, :N].astype(complex)
        C  = R[:N, N:].astype(complex)
        A2 = R[N:, N:].astype(complex)
        Ct = C.T
        assert np.allclose(Rh[:N, :N], 0.5*(A1+A2+C+Ct),   atol=1e-11)
        assert np.allclose(Rh[:N, N:], 0.5*(A1-A2-C+Ct),   atol=1e-11)
        assert np.allclose(Rh[N:, :N], 0.5*(A1-A2+C-Ct),   atol=1e-11)
        assert np.allclose(Rh[N:, N:], 0.5*(A1+A2-C-Ct),   atol=1e-11)

    def test_self_inverse(self, single):
        R, info = single
        Rhh = hadamard(hadamard(R, info), info)
        assert np.allclose(Rhh, R.astype(complex), atol=1e-11)

    def test_eigenspectrum(self, single):
        R, info = single
        assert np.allclose(eigs(R), eigs(hadamard(R, info)), atol=1e-10)

    def test_two_bit_eigenspectrum(self, two_bit):
        R, info = two_bit
        for q in (0, 1):
            assert np.allclose(eigs(R), eigs(hadamard(R, info, q)), atol=1e-10)

    def test_result_is_hermitian(self, single):
        R, info = single
        Rh = hadamard(R, info)
        assert np.allclose(Rh, Rh.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# Pauli-Y / T / Phase  (spectrum-only checks)
# ---------------------------------------------------------------------------

class TestOtherGates:

    def test_pauli_y_spectrum(self, single):
        R, info = single
        assert np.allclose(eigs(R), eigs(pauli_y(R, info)), atol=1e-10)

    def test_pauli_y_self_inverse(self, single):
        R, info = single
        assert np.allclose(pauli_y(pauli_y(R, info), info), R.astype(complex), atol=1e-11)

    def test_t_gate_spectrum(self, single):
        R, info = single
        assert np.allclose(eigs(R), eigs(t_gate(R, info)), atol=1e-10)

    def test_t8_identity(self, single):
        """T^8 = I (period of e^{iπ/4} is 8)."""
        R, info = single
        Rt = R.astype(complex)
        for _ in range(8):
            Rt = t_gate(Rt, info)
        assert np.allclose(Rt, R.astype(complex), atol=1e-10)

    def test_phase_zero_is_identity(self, single):
        R, info = single
        assert np.allclose(phase_gate(R, info, 0.0), R.astype(complex), atol=1e-12)

    def test_phase_pi4_matches_t(self, single):
        R, info = single
        assert np.allclose(phase_gate(R, info, np.pi/4), t_gate(R, info), atol=1e-12)

    def test_phase_spectrum(self, single):
        R, info = single
        assert np.allclose(eigs(R), eigs(phase_gate(R, info, np.pi/3)), atol=1e-10)


# ---------------------------------------------------------------------------
# Gate circuits
# ---------------------------------------------------------------------------

class TestCircuits:

    def test_hxh_equals_z(self, single):
        """H X H = Z."""
        R, info = single
        R_hxh = apply_gate_sequence(R, info, [('H', 0), ('X', 0), ('H', 0)])
        assert np.allclose(R_hxh, pauli_z(R, info), atol=1e-11)

    def test_hzh_equals_x(self, single):
        """H Z H = X."""
        R, info = single
        R_hzh = apply_gate_sequence(R, info, [('H', 0), ('Z', 0), ('H', 0)])
        assert np.allclose(R_hzh, pauli_x(R, info), atol=1e-11)

    def test_xx_identity(self, single):
        R, info = single
        assert np.allclose(
            apply_gate_sequence(R, info, [('X', 0), ('X', 0)]),
            R.astype(complex), atol=1e-12)

    def test_empty_is_identity(self, single):
        R, info = single
        assert np.allclose(
            apply_gate_sequence(R, info, []),
            R.astype(complex), atol=1e-12)

    def test_two_qubit_cancellation(self, two_bit):
        """X on q0, Z on q1, X on q0  ≡  Z on q1 alone."""
        R, info = two_bit
        R_seq = apply_gate_sequence(R, info, [('X', 0), ('Z', 1), ('X', 0)])
        R_z1  = pauli_z(R, info, target_qubit=1)
        assert np.allclose(R_seq, R_z1, atol=1e-11)

    def test_gates_on_different_qubits_commute(self, two_bit):
        """X on q0 and Z on q1 commute (act on independent subsystems)."""
        R, info = two_bit
        R1 = pauli_z(pauli_x(R, info, 0), info, 1)
        R2 = pauli_x(pauli_z(R, info, 1), info, 0)
        assert np.allclose(R1, R2, atol=1e-11)


# ---------------------------------------------------------------------------
# Gates on contracted / quotient matrices
# ---------------------------------------------------------------------------

from core.contraction import contract_ql_bit, minimal_quotient


class TestContractedGates:

    @pytest.fixture(scope="class")
    def partial(self):
        """Two-bit product with QL-bit 0 contracted."""
        a = generate_quantum_like_bit(N, K, L)
        b = generate_quantum_like_bit(N, K, L)
        coupled = couple_ql_bits(a, b)
        return contract_ql_bit(coupled, target=0)

    @pytest.fixture(scope="class")
    def full(self):
        """Two-bit product with both QL-bits contracted."""
        a = generate_quantum_like_bit(N, K, L)
        b = generate_quantum_like_bit(N, K, L)
        coupled = couple_ql_bits(a, b)
        step1 = contract_ql_bit(coupled, target=0)
        return contract_ql_bit(step1, target=1)

    @pytest.fixture(scope="class")
    def quotient(self):
        """Minimal quotient built directly (equivalent to full contraction)."""
        a = generate_quantum_like_bit(N, K, L)
        b = generate_quantum_like_bit(N, K, L)
        return minimal_quotient(a, b)

    # --- partial contraction ---

    def test_partial_gate_on_uncontracted_qubit_preserves_spectrum(self, partial):
        R, info = partial
        assert np.allclose(eigs(R), eigs(pauli_x(R, info, target_qubit=1)), atol=1e-10)

    def test_partial_gate_on_contracted_qubit_preserves_spectrum(self, partial):
        R, info = partial
        assert np.allclose(eigs(R), eigs(hadamard(R, info, target_qubit=0)), atol=1e-10)

    def test_partial_hxh_equals_z_on_contracted_qubit(self, partial):
        """H X H = Z still holds after contraction."""
        R, info = partial
        R_hxh = apply_gate_sequence(R, info, [('H', 0), ('X', 0), ('H', 0)])
        assert np.allclose(R_hxh, pauli_z(R, info, target_qubit=0), atol=1e-11)

    def test_partial_basis_change_shape(self, partial):
        R, info = partial
        U = compute_basis_change(info)
        assert U.shape == (R.shape[0], R.shape[0])

    def test_partial_basis_change_unitary(self, partial):
        R, info = partial
        U = compute_basis_change(info)
        n = U.shape[0]
        assert np.allclose(U @ U.conj().T, np.eye(n), atol=1e-11)

    # --- full contraction ---

    def test_full_gate_preserves_spectrum(self, full):
        R, info = full
        for q in (0, 1):
            assert np.allclose(eigs(R), eigs(pauli_x(R, info, target_qubit=q)), atol=1e-10)

    def test_full_hxh_equals_z(self, full):
        R, info = full
        R_hxh = apply_gate_sequence(R, info, [('H', 0), ('X', 0), ('H', 0)])
        assert np.allclose(R_hxh, pauli_z(R, info, target_qubit=0), atol=1e-11)

    # --- minimal quotient ---

    def test_quotient_gate_preserves_spectrum(self, quotient):
        R, info = quotient
        for q in (0, 1):
            assert np.allclose(eigs(R), eigs(hadamard(R, info, target_qubit=q)), atol=1e-10)

    def test_quotient_basis_change_unitary(self, quotient):
        R, info = quotient
        U = compute_basis_change(info)
        n = U.shape[0]
        assert np.allclose(U @ U.conj().T, np.eye(n), atol=1e-11)
