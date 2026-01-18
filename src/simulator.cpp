/**
 * @file simulator.cpp
 * @brief Quantum circuit simulator with noise and ZNE support
 */

#include "qmitigate/simulator.hpp"
#include <stdexcept>

namespace qmitigate {

Simulator::Simulator(NoiseModel noise_model)
    : noise_model_(std::move(noise_model)) {}

QuantumState Simulator::run(const Circuit &circuit) const {
  QuantumState state(circuit.num_qubits);
  return run(circuit, std::move(state));
}

QuantumState Simulator::run(const Circuit &circuit,
                            QuantumState initial_state) const {
  if (initial_state.num_qubits() != circuit.num_qubits) {
    throw std::invalid_argument(
        "Initial state qubit count doesn't match circuit");
  }

  for (const auto &gate : circuit.gates) {
    gates::apply_gate(initial_state, gate);
    if (noise_model_.has_noise()) {
      noise_model_.apply_noise(initial_state, gate, rng_);
    }
  }
  return initial_state;
}

Real Simulator::expectation_Z(const Circuit &circuit, QubitIndex qubit,
                              int shots) const {
  Real sum = 0.0;
  for (int i = 0; i < shots; ++i) {
    QuantumState state = run(circuit);
    sum += state.expectation_Z(qubit);
  }
  return sum / static_cast<Real>(shots);
}

std::vector<Real> Simulator::sample_expectation_Z(const Circuit &circuit,
                                                  QubitIndex qubit, int shots,
                                                  int samples) const {
  std::vector<Real> results;
  results.reserve(static_cast<std::size_t>(samples));
  for (int s = 0; s < samples; ++s) {
    results.push_back(expectation_Z(circuit, qubit, shots));
  }
  return results;
}

void Simulator::set_noise_scale(Real scale) {
  noise_model_.set_scale_factor(scale);
}

// =============================================================================
// Digital Folding for ZNE
// =============================================================================

Circuit fold_circuit_global(const Circuit &circuit, int scale_factor) {
  if (scale_factor < 1 || scale_factor % 2 == 0) {
    throw std::invalid_argument("Scale factor must be odd and >= 1");
  }

  if (scale_factor == 1) {
    return circuit; // No folding needed
  }

  Circuit folded(circuit.num_qubits);
  int num_folds = (scale_factor - 1) / 2;

  // Original circuit: U
  for (const auto &gate : circuit.gates) {
    folded.add_gate(gate);
  }

  // Append (U† U) for each fold
  for (int f = 0; f < num_folds; ++f) {
    // U† (inverse of each gate in reverse order)
    for (auto it = circuit.gates.rbegin(); it != circuit.gates.rend(); ++it) {
      GateInfo inverse = *it;
      // Negate rotation angles for inverse
      for (auto &param : inverse.parameters) {
        param = -param;
      }
      // For S, T gates we need different handling, but for now simplified
      folded.add_gate(inverse);
    }
    // U again
    for (const auto &gate : circuit.gates) {
      folded.add_gate(gate);
    }
  }

  return folded;
}

} // namespace qmitigate
