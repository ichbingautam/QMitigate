#pragma once

/**
 * @file types.hpp
 * @brief Core type definitions for the QMitigate quantum simulator
 *
 * This header defines the fundamental types used throughout the simulator,
 * including complex number types and precision settings.
 */

#include <complex>
#include <vector>
#include <cstdint>
#include <limits>

namespace qmitigate {

// =============================================================================
// Precision Configuration
// =============================================================================

/// Complex number type for quantum amplitudes
using Complex = std::complex<double>;

/// Real number type for probabilities and angles
using Real = double;

/// Size type for qubit indices and state vector dimensions
using QubitIndex = std::uint32_t;

/// State vector type
using StateVector = std::vector<Complex>;

// =============================================================================
// Constants
// =============================================================================

namespace constants {

/// Complex zero
inline constexpr Complex ZERO{0.0, 0.0};

/// Complex one
inline constexpr Complex ONE{1.0, 0.0};

/// Imaginary unit i
inline constexpr Complex I{0.0, 1.0};

/// 1/sqrt(2) for Hadamard gate
inline const Real INV_SQRT2 = 1.0 / std::sqrt(2.0);

/// Pi constant
inline constexpr Real PI = 3.14159265358979323846;

/// Numerical tolerance for floating point comparisons
inline constexpr Real TOLERANCE = 1e-10;

/// Maximum number of qubits (limited by memory: 2^30 * 16 bytes ≈ 17 GB)
inline constexpr QubitIndex MAX_QUBITS = 30;

} // namespace constants

// =============================================================================
// Gate Types
// =============================================================================

/**
 * @enum GateType
 * @brief Enumeration of supported quantum gate types
 */
enum class GateType {
    // Single-qubit gates
    Identity,   ///< Identity gate I
    PauliX,     ///< Pauli-X (NOT) gate
    PauliY,     ///< Pauli-Y gate
    PauliZ,     ///< Pauli-Z gate
    Hadamard,   ///< Hadamard gate H
    S,          ///< S (Phase) gate
    T,          ///< T gate (π/8)
    SDag,       ///< S-dagger gate
    TDag,       ///< T-dagger gate

    // Rotation gates
    RX,         ///< Rotation around X-axis
    RY,         ///< Rotation around Y-axis
    RZ,         ///< Rotation around Z-axis

    // Two-qubit gates
    CNOT,       ///< Controlled-NOT (CX)
    CZ,         ///< Controlled-Z
    SWAP,       ///< SWAP gate
    CY,         ///< Controlled-Y

    // Three-qubit gates
    Toffoli,    ///< Toffoli (CCX) gate
    Fredkin,    ///< Fredkin (CSWAP) gate

    // Measurement
    Measure     ///< Computational basis measurement
};

/**
 * @enum NoiseChannel
 * @brief Types of noise channels for error simulation
 */
enum class NoiseChannel {
    None,           ///< No noise (ideal)
    Depolarizing,   ///< Depolarizing channel
    BitFlip,        ///< Bit-flip channel (X errors)
    PhaseFlip,      ///< Phase-flip channel (Z errors)
    BitPhaseFlip,   ///< Bit-phase-flip channel (Y errors)
    AmplitudeDamping, ///< Amplitude damping (T1)
    PhaseDamping    ///< Phase damping (T2)
};

// =============================================================================
// Gate Information Structure
// =============================================================================

/**
 * @struct GateInfo
 * @brief Metadata about a quantum gate operation
 */
struct GateInfo {
    GateType type;
    std::vector<QubitIndex> targets;
    std::vector<QubitIndex> controls;
    std::vector<Real> parameters;

    GateInfo(GateType gate_type, std::vector<QubitIndex> target_qubits)
        : type(gate_type), targets(std::move(target_qubits)) {}

    GateInfo(GateType gate_type,
             std::vector<QubitIndex> target_qubits,
             std::vector<QubitIndex> control_qubits)
        : type(gate_type),
          targets(std::move(target_qubits)),
          controls(std::move(control_qubits)) {}

    GateInfo(GateType gate_type,
             std::vector<QubitIndex> target_qubits,
             std::vector<Real> params)
        : type(gate_type),
          targets(std::move(target_qubits)),
          parameters(std::move(params)) {}
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Calculate the state vector dimension for n qubits
 * @param num_qubits Number of qubits
 * @return 2^num_qubits (the dimension of the Hilbert space)
 */
inline constexpr std::size_t hilbert_dimension(QubitIndex num_qubits) {
    return static_cast<std::size_t>(1) << num_qubits;
}

/**
 * @brief Check if the k-th bit of index n is set
 * @param n The index
 * @param k The bit position
 * @return true if bit k is 1
 */
inline constexpr bool bit_is_set(std::size_t n, QubitIndex k) {
    return (n >> k) & 1;
}

/**
 * @brief Set the k-th bit of n to value
 * @param n The index
 * @param k The bit position
 * @param value The bit value (0 or 1)
 * @return Modified index
 */
inline constexpr std::size_t set_bit(std::size_t n, QubitIndex k, bool value) {
    if (value) {
        return n | (static_cast<std::size_t>(1) << k);
    } else {
        return n & ~(static_cast<std::size_t>(1) << k);
    }
}

/**
 * @brief Flip the k-th bit of n
 * @param n The index
 * @param k The bit position
 * @return Modified index with bit k flipped
 */
inline constexpr std::size_t flip_bit(std::size_t n, QubitIndex k) {
    return n ^ (static_cast<std::size_t>(1) << k);
}

} // namespace qmitigate
