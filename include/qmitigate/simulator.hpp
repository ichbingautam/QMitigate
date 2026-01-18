#pragma once

#include "qmitigate/gates.hpp"
#include "qmitigate/noise_model.hpp"
#include "qmitigate/quantum_state.hpp"
#include "qmitigate/types.hpp"
#include <random>
#include <vector>

namespace qmitigate {

struct Circuit {
  QubitIndex num_qubits;
  std::vector<GateInfo> gates;

  explicit Circuit(QubitIndex n) : num_qubits(n) {}

  void add_gate(const GateInfo &gate) { gates.push_back(gate); }
  void h(QubitIndex q) {
    gates.emplace_back(GateType::Hadamard, std::vector<QubitIndex>{q});
  }
  void x(QubitIndex q) {
    gates.emplace_back(GateType::PauliX, std::vector<QubitIndex>{q});
  }
  void y(QubitIndex q) {
    gates.emplace_back(GateType::PauliY, std::vector<QubitIndex>{q});
  }
  void z(QubitIndex q) {
    gates.emplace_back(GateType::PauliZ, std::vector<QubitIndex>{q});
  }
  void cnot(QubitIndex ctrl, QubitIndex tgt) {
    gates.emplace_back(GateType::CNOT, std::vector<QubitIndex>{tgt},
                       std::vector<QubitIndex>{ctrl});
  }
  void rx(QubitIndex q, Real theta) {
    gates.emplace_back(GateType::RX, std::vector<QubitIndex>{q},
                       std::vector<Real>{theta});
  }
  void ry(QubitIndex q, Real theta) {
    gates.emplace_back(GateType::RY, std::vector<QubitIndex>{q},
                       std::vector<Real>{theta});
  }
  void rz(QubitIndex q, Real theta) {
    gates.emplace_back(GateType::RZ, std::vector<QubitIndex>{q},
                       std::vector<Real>{theta});
  }
  [[nodiscard]] std::size_t depth() const { return gates.size(); }
};

class Simulator {
public:
  Simulator() = default;
  explicit Simulator(NoiseModel noise_model);

  [[nodiscard]] QuantumState run(const Circuit &circuit) const;
  [[nodiscard]] QuantumState run(const Circuit &circuit,
                                 QuantumState initial_state) const;

  [[nodiscard]] Real expectation_Z(const Circuit &circuit, QubitIndex qubit,
                                   int shots = 1000) const;
  [[nodiscard]] std::vector<Real> sample_expectation_Z(const Circuit &circuit,
                                                       QubitIndex qubit,
                                                       int shots,
                                                       int samples) const;

  void set_noise_model(NoiseModel model) { noise_model_ = std::move(model); }
  void set_noise_scale(Real scale);
  [[nodiscard]] Real noise_scale() const { return noise_model_.scale_factor(); }
  void seed(unsigned int s) { noise_model_.seed(s); }

private:
  mutable NoiseModel noise_model_;
  mutable std::mt19937 rng_{std::random_device{}()};
};

// Digital folding for ZNE
[[nodiscard]] Circuit fold_circuit_global(const Circuit &circuit,
                                          int scale_factor);

} // namespace qmitigate
