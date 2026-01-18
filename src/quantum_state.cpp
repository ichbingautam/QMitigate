/**
 * @file quantum_state.cpp
 * @brief Implementation of QuantumState class
 */

#include "qmitigate/quantum_state.hpp"
#include <bitset>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace qmitigate {

// =============================================================================
// Constructors
// =============================================================================

QuantumState::QuantumState(QubitIndex num_qubits)
    : num_qubits_(num_qubits),
      state_(hilbert_dimension(num_qubits), constants::ZERO) {
  if (num_qubits > constants::MAX_QUBITS) {
    throw std::invalid_argument("Number of qubits exceeds maximum (" +
                                std::to_string(constants::MAX_QUBITS) + ")");
  }
  if (num_qubits == 0) {
    throw std::invalid_argument("Number of qubits must be positive");
  }
  // Initialize to |00...0⟩
  state_[0] = constants::ONE;
}

QuantumState::QuantumState(StateVector state_vector)
    : state_(std::move(state_vector)) {
  // Check that size is a power of 2
  std::size_t size = state_.size();
  if (size == 0 || (size & (size - 1)) != 0) {
    throw std::invalid_argument("State vector size must be a power of 2");
  }
  // Calculate number of qubits
  num_qubits_ = 0;
  while ((static_cast<std::size_t>(1) << num_qubits_) < size) {
    ++num_qubits_;
  }
}

// =============================================================================
// State Initialization
// =============================================================================

void QuantumState::reset() {
#pragma omp parallel for
  for (std::size_t i = 0; i < state_.size(); ++i) {
    state_[i] = constants::ZERO;
  }
  state_[0] = constants::ONE;
}

void QuantumState::initialize_basis_state(std::size_t basis_state) {
  if (basis_state >= state_.size()) {
    throw std::out_of_range("Basis state index out of range");
  }
#pragma omp parallel for
  for (std::size_t i = 0; i < state_.size(); ++i) {
    state_[i] = constants::ZERO;
  }
  state_[basis_state] = constants::ONE;
}

void QuantumState::initialize_superposition() {
  Real amp = 1.0 / std::sqrt(static_cast<Real>(state_.size()));
#pragma omp parallel for
  for (std::size_t i = 0; i < state_.size(); ++i) {
    state_[i] = Complex(amp, 0.0);
  }
}

// =============================================================================
// State Properties
// =============================================================================

Real QuantumState::norm() const {
  Real sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (std::size_t i = 0; i < state_.size(); ++i) {
    sum += std::norm(state_[i]);
  }
  return std::sqrt(sum);
}

void QuantumState::normalize() {
  Real n = norm();
  if (n < constants::TOLERANCE) {
    throw std::runtime_error("Cannot normalize zero state");
  }
  Real inv_norm = 1.0 / n;
#pragma omp parallel for
  for (std::size_t i = 0; i < state_.size(); ++i) {
    state_[i] *= inv_norm;
  }
}

bool QuantumState::is_normalized() const {
  return std::abs(norm() - 1.0) < constants::TOLERANCE;
}

Real QuantumState::fidelity(const QuantumState &other) const {
  Complex ip = inner_product(other);
  return std::norm(ip); // |⟨ψ|φ⟩|²
}

Complex QuantumState::inner_product(const QuantumState &other) const {
  if (state_.size() != other.state_.size()) {
    throw std::invalid_argument("States must have same dimension");
  }
  Complex sum{0.0, 0.0};
#pragma omp parallel for reduction(+ : sum)
  for (std::size_t i = 0; i < state_.size(); ++i) {
    sum += std::conj(state_[i]) * other.state_[i];
  }
  return sum;
}

// =============================================================================
// Measurement
// =============================================================================

std::size_t QuantumState::measure(std::mt19937 &rng) {
  std::uniform_real_distribution<Real> dist(0.0, 1.0);
  Real r = dist(rng);
  Real cumsum = 0.0;

  for (std::size_t i = 0; i < state_.size(); ++i) {
    cumsum += std::norm(state_[i]);
    if (r <= cumsum) {
// Collapse to this state
#pragma omp parallel for
      for (std::size_t j = 0; j < state_.size(); ++j) {
        state_[j] = (j == i) ? constants::ONE : constants::ZERO;
      }
      return i;
    }
  }
  return state_.size() - 1;
}

int QuantumState::measure_qubit(QubitIndex qubit, std::mt19937 &rng) {
  Real p0 = probability_zero(qubit);
  std::uniform_real_distribution<Real> dist(0.0, 1.0);
  int result = (dist(rng) < p0) ? 0 : 1;

  // Collapse the state
  Real norm_factor = 1.0 / std::sqrt(result == 0 ? p0 : (1.0 - p0));

#pragma omp parallel for
  for (std::size_t i = 0; i < state_.size(); ++i) {
    bool bit = bit_is_set(i, qubit);
    if ((result == 0 && bit) || (result == 1 && !bit)) {
      state_[i] = constants::ZERO;
    } else {
      state_[i] *= norm_factor;
    }
  }
  return result;
}

Real QuantumState::probability_zero(QubitIndex qubit) const {
  Real p0 = 0.0;
#pragma omp parallel for reduction(+ : p0)
  for (std::size_t i = 0; i < state_.size(); ++i) {
    if (!bit_is_set(i, qubit)) {
      p0 += std::norm(state_[i]);
    }
  }
  return p0;
}

std::vector<Real> QuantumState::get_probabilities() const {
  std::vector<Real> probs(state_.size());
#pragma omp parallel for
  for (std::size_t i = 0; i < state_.size(); ++i) {
    probs[i] = std::norm(state_[i]);
  }
  return probs;
}

// =============================================================================
// Expectation Values
// =============================================================================

Real QuantumState::expectation_Z(QubitIndex qubit) const {
  Real p0 = probability_zero(qubit);
  return 2.0 * p0 - 1.0; // ⟨Z⟩ = P(0) - P(1) = 2P(0) - 1
}

Real QuantumState::expectation_ZZ(const std::vector<QubitIndex> &qubits) const {
  Real expectation = 0.0;
#pragma omp parallel for reduction(+ : expectation)
  for (std::size_t i = 0; i < state_.size(); ++i) {
    int parity = 0;
    for (QubitIndex q : qubits) {
      if (bit_is_set(i, q))
        parity ^= 1;
    }
    Real sign = (parity == 0) ? 1.0 : -1.0;
    expectation += sign * std::norm(state_[i]);
  }
  return expectation;
}

// =============================================================================
// Utility
// =============================================================================

std::string QuantumState::to_string(Real threshold) const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  bool first = true;

  for (std::size_t i = 0; i < state_.size(); ++i) {
    if (std::abs(state_[i]) > threshold) {
      if (!first)
        oss << " + ";
      first = false;
      oss << "(" << state_[i].real();
      if (state_[i].imag() >= 0)
        oss << "+";
      oss << state_[i].imag() << "i)|";
      for (QubitIndex q = num_qubits_; q > 0; --q) {
        oss << (bit_is_set(i, q - 1) ? "1" : "0");
      }
      oss << "⟩";
    }
  }
  return oss.str();
}

// =============================================================================
// Factory Functions
// =============================================================================

QuantumState create_bell_state() {
  StateVector sv(4, constants::ZERO);
  sv[0] = Complex(constants::INV_SQRT2, 0.0);
  sv[3] = Complex(constants::INV_SQRT2, 0.0);
  return QuantumState(std::move(sv));
}

QuantumState create_ghz_state(QubitIndex num_qubits) {
  if (num_qubits < 2) {
    throw std::invalid_argument("GHZ state requires at least 2 qubits");
  }
  StateVector sv(hilbert_dimension(num_qubits), constants::ZERO);
  Real amp = constants::INV_SQRT2;
  sv[0] = Complex(amp, 0.0);
  sv[sv.size() - 1] = Complex(amp, 0.0);
  return QuantumState(std::move(sv));
}

QuantumState create_w_state(QubitIndex num_qubits) {
  if (num_qubits < 2) {
    throw std::invalid_argument("W state requires at least 2 qubits");
  }
  StateVector sv(hilbert_dimension(num_qubits), constants::ZERO);
  Real amp = 1.0 / std::sqrt(static_cast<Real>(num_qubits));
  for (QubitIndex i = 0; i < num_qubits; ++i) {
    sv[static_cast<std::size_t>(1) << i] = Complex(amp, 0.0);
  }
  return QuantumState(std::move(sv));
}

} // namespace qmitigate
