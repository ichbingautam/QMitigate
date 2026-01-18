#pragma once

/**
 * @file quantum_state.hpp
 * @brief QuantumState class for representing and manipulating quantum states
 *
 * The quantum state of N qubits is represented as a complex vector of dimension
 * 2^N. This class provides efficient methods for state manipulation and
 * measurement.
 *
 * Mathematical Background:
 * - A pure quantum state |ψ⟩ = Σ αᵢ|i⟩ where i ∈ {0, 1}^N
 * - The amplitudes αᵢ satisfy the normalization condition: Σ|αᵢ|² = 1
 * - Probabilities are computed as P(i) = |αᵢ|²
 */

#include "qmitigate/types.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

#ifdef QMITIGATE_USE_OPENMP
#include <omp.h>
#endif

namespace qmitigate {

/**
 * @class QuantumState
 * @brief Represents a quantum state vector for N qubits
 *
 * This class implements a state-vector simulator optimized for:
 * - Cache-efficient memory access patterns
 * - OpenMP parallelization for large state vectors
 * - Numerical stability in normalization
 */
class QuantumState {
public:
  // =========================================================================
  // Constructors
  // =========================================================================

  /**
   * @brief Construct a quantum state initialized to |00...0⟩
   * @param num_qubits Number of qubits (must be ≤ MAX_QUBITS)
   * @throws std::invalid_argument if num_qubits > MAX_QUBITS
   */
  explicit QuantumState(QubitIndex num_qubits);

  /**
   * @brief Construct a quantum state from an existing state vector
   * @param state_vector Initial state (must have size 2^n for some n)
   * @throws std::invalid_argument if state_vector size is not a power of 2
   */
  explicit QuantumState(StateVector state_vector);

  /**
   * @brief Copy constructor
   */
  QuantumState(const QuantumState &other) = default;

  /**
   * @brief Move constructor
   */
  QuantumState(QuantumState &&other) noexcept = default;

  /**
   * @brief Copy assignment
   */
  QuantumState &operator=(const QuantumState &other) = default;

  /**
   * @brief Move assignment
   */
  QuantumState &operator=(QuantumState &&other) noexcept = default;

  /**
   * @brief Destructor
   */
  ~QuantumState() = default;

  // =========================================================================
  // State Initialization
  // =========================================================================

  /**
   * @brief Reset state to |00...0⟩
   */
  void reset();

  /**
   * @brief Initialize to a specific computational basis state |i⟩
   * @param basis_state The basis state index (binary representation)
   * @throws std::out_of_range if basis_state >= dimension
   */
  void initialize_basis_state(std::size_t basis_state);

  /**
   * @brief Create a uniform superposition over all basis states
   *
   * Creates the state |+⟩⊗N = (1/√2^N) Σ|i⟩
   */
  void initialize_superposition();

  // =========================================================================
  // Accessors
  // =========================================================================

  /**
   * @brief Get the number of qubits
   * @return Number of qubits
   */
  [[nodiscard]] QubitIndex num_qubits() const noexcept { return num_qubits_; }

  /**
   * @brief Get the state vector dimension (2^num_qubits)
   * @return Hilbert space dimension
   */
  [[nodiscard]] std::size_t dimension() const noexcept { return state_.size(); }

  /**
   * @brief Get the amplitude at index i
   * @param i Basis state index
   * @return Complex amplitude αᵢ
   */
  [[nodiscard]] Complex amplitude(std::size_t i) const { return state_.at(i); }

  /**
   * @brief Get the probability of measuring basis state i
   * @param i Basis state index
   * @return Probability P(i) = |αᵢ|²
   */
  [[nodiscard]] Real probability(std::size_t i) const {
    return std::norm(state_.at(i));
  }

  /**
   * @brief Get read-only access to the underlying state vector
   * @return Const reference to state vector
   */
  [[nodiscard]] const StateVector &state_vector() const noexcept {
    return state_;
  }

  /**
   * @brief Get mutable access to the state vector (use with caution)
   * @return Reference to state vector
   */
  [[nodiscard]] StateVector &state_vector() noexcept { return state_; }

  // =========================================================================
  // State Properties
  // =========================================================================

  /**
   * @brief Calculate the norm of the state vector
   * @return √(Σ|αᵢ|²) - should be 1.0 for a valid quantum state
   */
  [[nodiscard]] Real norm() const;

  /**
   * @brief Normalize the state vector to unit norm
   */
  void normalize();

  /**
   * @brief Check if the state is normalized (within tolerance)
   * @return true if |norm - 1| < TOLERANCE
   */
  [[nodiscard]] bool is_normalized() const;

  /**
   * @brief Calculate the state fidelity with another state
   *
   * Fidelity F(ρ, σ) = |⟨ψ|φ⟩|² for pure states
   *
   * @param other The other quantum state
   * @return Fidelity value in [0, 1]
   */
  [[nodiscard]] Real fidelity(const QuantumState &other) const;

  /**
   * @brief Calculate the inner product ⟨this|other⟩
   * @param other The other quantum state
   * @return Complex inner product
   */
  [[nodiscard]] Complex inner_product(const QuantumState &other) const;

  // =========================================================================
  // Measurement
  // =========================================================================

  /**
   * @brief Perform a projective measurement in the computational basis
   *
   * This collapses the state to a single basis state according to the
   * Born rule probability distribution.
   *
   * @param rng Random number generator
   * @return Measurement outcome (basis state index)
   */
  std::size_t measure(std::mt19937 &rng);

  /**
   * @brief Measure a single qubit in the computational basis
   *
   * Partially collapses the state and returns the measurement outcome.
   *
   * @param qubit The qubit index to measure
   * @param rng Random number generator
   * @return Measurement outcome (0 or 1)
   */
  int measure_qubit(QubitIndex qubit, std::mt19937 &rng);

  /**
   * @brief Get the probability of measuring |0⟩ on a specific qubit
   * @param qubit The qubit index
   * @return Probability P(qubit = 0)
   */
  [[nodiscard]] Real probability_zero(QubitIndex qubit) const;

  /**
   * @brief Get all measurement probabilities
   * @return Vector of probabilities for each basis state
   */
  [[nodiscard]] std::vector<Real> get_probabilities() const;

  // =========================================================================
  // Expectation Values
  // =========================================================================

  /**
   * @brief Calculate expectation value of Z operator on a qubit
   *
   * ⟨Z⟩ = P(0) - P(1)
   *
   * @param qubit The qubit index
   * @return Expectation value in [-1, 1]
   */
  [[nodiscard]] Real expectation_Z(QubitIndex qubit) const;

  /**
   * @brief Calculate expectation value of a Pauli Z string
   *
   * For operator O = Z_q1 ⊗ Z_q2 ⊗ ... ⊗ Z_qn
   *
   * @param qubits Vector of qubit indices
   * @return Expectation value
   */
  [[nodiscard]] Real
  expectation_ZZ(const std::vector<QubitIndex> &qubits) const;

  // =========================================================================
  // Utility
  // =========================================================================

  /**
   * @brief Get a string representation of the state
   * @param threshold Only show amplitudes with magnitude above this
   * @return Formatted string representation
   */
  [[nodiscard]] std::string to_string(Real threshold = 1e-6) const;

private:
  QubitIndex num_qubits_; ///< Number of qubits
  StateVector state_;     ///< State vector amplitudes
};

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * @brief Create a Bell state (|00⟩ + |11⟩)/√2
 * @return Two-qubit Bell state
 */
[[nodiscard]] QuantumState create_bell_state();

/**
 * @brief Create a GHZ state (|00...0⟩ + |11...1⟩)/√2
 * @param num_qubits Number of qubits (must be ≥ 2)
 * @return N-qubit GHZ state
 */
[[nodiscard]] QuantumState create_ghz_state(QubitIndex num_qubits);

/**
 * @brief Create a W state for N qubits
 *
 * W state = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √N
 *
 * @param num_qubits Number of qubits (must be ≥ 2)
 * @return N-qubit W state
 */
[[nodiscard]] QuantumState create_w_state(QubitIndex num_qubits);

} // namespace qmitigate
