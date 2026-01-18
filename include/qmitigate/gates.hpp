#pragma once

/**
 * @file gates.hpp
 * @brief Quantum gate operations optimized for state-vector simulation
 *
 * This module implements quantum gates using the efficient "sparse" application
 * method. Instead of constructing full 2^N × 2^N matrices, gates are applied
 * by iterating over pairs of state vector indices.
 *
 * Key Optimization:
 * - Standard matrix multiplication: O(4^N) complexity
 * - Our implementation: O(2^N) complexity
 *
 * This is crucial for scaling to 20+ qubits.
 *
 * Mathematical Reference:
 * For a single-qubit gate U on qubit k, we find all pairs (i, j) where:
 * - Bit k of i is 0, bit k of j is 1
 * - All other bits are identical
 * Then apply: [state[i]', state[j]']^T = U × [state[i], state[j]]^T
 */

#include "qmitigate/quantum_state.hpp"
#include "qmitigate/types.hpp"
#include <cmath>
#include <omp.h>

namespace qmitigate {
namespace gates {

// =============================================================================
// Single-Qubit Gates
// =============================================================================

/**
 * @brief Apply the Pauli-X (NOT) gate
 *
 * X = [0 1]
 *     [1 0]
 *
 * Action: |0⟩ → |1⟩, |1⟩ → |0⟩
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_X(QuantumState &state, QubitIndex qubit);

/**
 * @brief Apply the Pauli-Y gate
 *
 * Y = [0 -i]
 *     [i  0]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_Y(QuantumState &state, QubitIndex qubit);

/**
 * @brief Apply the Pauli-Z gate
 *
 * Z = [1  0]
 *     [0 -1]
 *
 * Action: |0⟩ → |0⟩, |1⟩ → -|1⟩
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_Z(QuantumState &state, QubitIndex qubit);

/**
 * @brief Apply the Hadamard gate
 *
 * H = (1/√2) [1  1]
 *            [1 -1]
 *
 * Action: |0⟩ → |+⟩ = (|0⟩+|1⟩)/√2
 *         |1⟩ → |−⟩ = (|0⟩-|1⟩)/√2
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_H(QuantumState &state, QubitIndex qubit);

/**
 * @brief Apply the S (Phase) gate
 *
 * S = [1 0]
 *     [0 i]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_S(QuantumState &state, QubitIndex qubit);

/**
 * @brief Apply the T gate (π/8 gate)
 *
 * T = [1    0   ]
 *     [0 e^(iπ/4)]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_T(QuantumState &state, QubitIndex qubit);

/**
 * @brief Apply the S-dagger gate
 *
 * S† = [1  0]
 *      [0 -i]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_Sdag(QuantumState &state, QubitIndex qubit);

/**
 * @brief Apply the T-dagger gate
 *
 * T† = [1     0    ]
 *      [0 e^(-iπ/4)]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 */
void apply_Tdag(QuantumState &state, QubitIndex qubit);

// =============================================================================
// Rotation Gates
// =============================================================================

/**
 * @brief Apply rotation around X-axis
 *
 * RX(θ) = [cos(θ/2)   -i·sin(θ/2)]
 *         [-i·sin(θ/2)  cos(θ/2) ]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 * @param theta Rotation angle in radians
 */
void apply_RX(QuantumState &state, QubitIndex qubit, Real theta);

/**
 * @brief Apply rotation around Y-axis
 *
 * RY(θ) = [cos(θ/2)  -sin(θ/2)]
 *         [sin(θ/2)   cos(θ/2)]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 * @param theta Rotation angle in radians
 */
void apply_RY(QuantumState &state, QubitIndex qubit, Real theta);

/**
 * @brief Apply rotation around Z-axis
 *
 * RZ(θ) = [e^(-iθ/2)    0    ]
 *         [   0      e^(iθ/2)]
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 * @param theta Rotation angle in radians
 */
void apply_RZ(QuantumState &state, QubitIndex qubit, Real theta);

/**
 * @brief Apply a general single-qubit rotation U3(θ, φ, λ)
 *
 * U3 = [cos(θ/2)         -e^(iλ)·sin(θ/2)    ]
 *      [e^(iφ)·sin(θ/2)   e^(i(φ+λ))·cos(θ/2)]
 *
 * This is the most general single-qubit gate.
 *
 * @param state The quantum state to modify
 * @param qubit Target qubit index
 * @param theta Zenith angle
 * @param phi Azimuth angle
 * @param lambda Phase angle
 */
void apply_U3(QuantumState &state, QubitIndex qubit, Real theta, Real phi,
              Real lambda);

// =============================================================================
// Two-Qubit Gates
// =============================================================================

/**
 * @brief Apply CNOT (Controlled-NOT) gate
 *
 * Flips target qubit if control qubit is |1⟩
 *
 * |00⟩ → |00⟩
 * |01⟩ → |01⟩
 * |10⟩ → |11⟩
 * |11⟩ → |10⟩
 *
 * @param state The quantum state to modify
 * @param control Control qubit index
 * @param target Target qubit index
 */
void apply_CNOT(QuantumState &state, QubitIndex control, QubitIndex target);

/**
 * @brief Apply CZ (Controlled-Z) gate
 *
 * Applies Z to target if control is |1⟩
 *
 * @param state The quantum state to modify
 * @param control Control qubit index
 * @param target Target qubit index
 */
void apply_CZ(QuantumState &state, QubitIndex control, QubitIndex target);

/**
 * @brief Apply CY (Controlled-Y) gate
 *
 * @param state The quantum state to modify
 * @param control Control qubit index
 * @param target Target qubit index
 */
void apply_CY(QuantumState &state, QubitIndex control, QubitIndex target);

/**
 * @brief Apply SWAP gate
 *
 * Swaps the states of two qubits
 *
 * @param state The quantum state to modify
 * @param qubit1 First qubit index
 * @param qubit2 Second qubit index
 */
void apply_SWAP(QuantumState &state, QubitIndex qubit1, QubitIndex qubit2);

/**
 * @brief Apply Controlled-RZ (CRZ) gate
 *
 * @param state The quantum state to modify
 * @param control Control qubit index
 * @param target Target qubit index
 * @param theta Rotation angle
 */
void apply_CRZ(QuantumState &state, QubitIndex control, QubitIndex target,
               Real theta);

// =============================================================================
// Three-Qubit Gates
// =============================================================================

/**
 * @brief Apply Toffoli (CCX) gate
 *
 * Flips target if both controls are |1⟩
 *
 * @param state The quantum state to modify
 * @param control1 First control qubit index
 * @param control2 Second control qubit index
 * @param target Target qubit index
 */
void apply_Toffoli(QuantumState &state, QubitIndex control1,
                   QubitIndex control2, QubitIndex target);

/**
 * @brief Apply Fredkin (CSWAP) gate
 *
 * Swaps two qubits if control is |1⟩
 *
 * @param state The quantum state to modify
 * @param control Control qubit index
 * @param target1 First target qubit index
 * @param target2 Second target qubit index
 */
void apply_Fredkin(QuantumState &state, QubitIndex control, QubitIndex target1,
                   QubitIndex target2);

// =============================================================================
// Multi-Qubit Controlled Gates
// =============================================================================

/**
 * @brief Apply a multi-controlled X gate
 *
 * @param state The quantum state to modify
 * @param controls Vector of control qubit indices
 * @param target Target qubit index
 */
void apply_MCX(QuantumState &state, const std::vector<QubitIndex> &controls,
               QubitIndex target);

// =============================================================================
// Gate Application by Type
// =============================================================================

/**
 * @brief Apply a gate specified by GateInfo
 *
 * This is a dispatcher function that routes to the appropriate gate
 * implementation.
 *
 * @param state The quantum state to modify
 * @param gate Gate information structure
 */
void apply_gate(QuantumState &state, const GateInfo &gate);

/**
 * @brief Apply the inverse (adjoint) of a gate
 *
 * For unitary gates, U^† U = I
 *
 * @param state The quantum state to modify
 * @param gate Gate information structure
 */
void apply_gate_inverse(QuantumState &state, const GateInfo &gate);

} // namespace gates
} // namespace qmitigate
