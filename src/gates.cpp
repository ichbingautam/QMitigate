/**
 * @file gates.cpp
 * @brief Optimized quantum gate implementations
 *
 * Key optimization: O(2^N) sparse application instead of O(4^N) matrix
 * multiplication
 */

#include "qmitigate/gates.hpp"
#include <stdexcept>

namespace qmitigate {
namespace gates {

// =============================================================================
// Single-Qubit Gates
// =============================================================================

void apply_X(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i; j < i + step; ++j) {
      std::swap(sv[j], sv[j + step]);
    }
  }
}

void apply_Y(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i; j < i + step; ++j) {
      Complex a = sv[j];
      Complex b = sv[j + step];
      sv[j] = Complex(b.imag(), -b.real());        // -i * b
      sv[j + step] = Complex(-a.imag(), a.real()); // i * a
    }
  }
}

void apply_Z(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i + step; j < i + 2 * step; ++j) {
      sv[j] = -sv[j];
    }
  }
}

void apply_H(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;
  const Real factor = constants::INV_SQRT2;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i; j < i + step; ++j) {
      Complex a = sv[j];
      Complex b = sv[j + step];
      sv[j] = factor * (a + b);
      sv[j + step] = factor * (a - b);
    }
  }
}

void apply_S(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i + step; j < i + 2 * step; ++j) {
      sv[j] *= constants::I;
    }
  }
}

void apply_T(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;
  const Complex phase = std::exp(constants::I * constants::PI / 4.0);

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i + step; j < i + 2 * step; ++j) {
      sv[j] *= phase;
    }
  }
}

void apply_Sdag(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i + step; j < i + 2 * step; ++j) {
      sv[j] *= -constants::I;
    }
  }
}

void apply_Tdag(QuantumState &state, QubitIndex qubit) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;
  const Complex phase = std::exp(-constants::I * constants::PI / 4.0);

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i + step; j < i + 2 * step; ++j) {
      sv[j] *= phase;
    }
  }
}

// =============================================================================
// Rotation Gates
// =============================================================================

void apply_RX(QuantumState &state, QubitIndex qubit, Real theta) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;
  const Real c = std::cos(theta / 2.0);
  const Real s = std::sin(theta / 2.0);

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i; j < i + step; ++j) {
      Complex a = sv[j];
      Complex b = sv[j + step];
      sv[j] = c * a - constants::I * s * b;
      sv[j + step] = -constants::I * s * a + c * b;
    }
  }
}

void apply_RY(QuantumState &state, QubitIndex qubit, Real theta) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;
  const Real c = std::cos(theta / 2.0);
  const Real s = std::sin(theta / 2.0);

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i; j < i + step; ++j) {
      Complex a = sv[j];
      Complex b = sv[j + step];
      sv[j] = c * a - s * b;
      sv[j + step] = s * a + c * b;
    }
  }
}

void apply_RZ(QuantumState &state, QubitIndex qubit, Real theta) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;
  const Complex phase_neg = std::exp(-constants::I * theta / 2.0);
  const Complex phase_pos = std::exp(constants::I * theta / 2.0);

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i; j < i + step; ++j) {
      sv[j] *= phase_neg;
      sv[j + step] *= phase_pos;
    }
  }
}

void apply_U3(QuantumState &state, QubitIndex qubit, Real theta, Real phi,
              Real lambda) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t step = static_cast<std::size_t>(1) << qubit;

  const Real c = std::cos(theta / 2.0);
  const Real s = std::sin(theta / 2.0);
  const Complex exp_phi = std::exp(constants::I * phi);
  const Complex exp_lam = std::exp(constants::I * lambda);
  const Complex exp_sum = std::exp(constants::I * (phi + lambda));

#pragma omp parallel for
  for (std::size_t i = 0; i < N; i += 2 * step) {
    for (std::size_t j = i; j < i + step; ++j) {
      Complex a = sv[j];
      Complex b = sv[j + step];
      sv[j] = c * a - exp_lam * s * b;
      sv[j + step] = exp_phi * s * a + exp_sum * c * b;
    }
  }
}

// =============================================================================
// Two-Qubit Gates
// =============================================================================

void apply_CNOT(QuantumState &state, QubitIndex control, QubitIndex target) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t ctrl_mask = static_cast<std::size_t>(1) << control;
  const std::size_t tgt_mask = static_cast<std::size_t>(1) << target;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    if ((i & ctrl_mask) && !(i & tgt_mask)) {
      std::size_t j = i ^ tgt_mask;
      std::swap(sv[i], sv[j]);
    }
  }
}

void apply_CZ(QuantumState &state, QubitIndex control, QubitIndex target) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t ctrl_mask = static_cast<std::size_t>(1) << control;
  const std::size_t tgt_mask = static_cast<std::size_t>(1) << target;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    if ((i & ctrl_mask) && (i & tgt_mask)) {
      sv[i] = -sv[i];
    }
  }
}

void apply_CY(QuantumState &state, QubitIndex control, QubitIndex target) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t ctrl_mask = static_cast<std::size_t>(1) << control;
  const std::size_t tgt_mask = static_cast<std::size_t>(1) << target;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    if ((i & ctrl_mask) && !(i & tgt_mask)) {
      std::size_t j = i ^ tgt_mask;
      Complex a = sv[i];
      Complex b = sv[j];
      sv[i] = Complex(b.imag(), -b.real());
      sv[j] = Complex(-a.imag(), a.real());
    }
  }
}

void apply_SWAP(QuantumState &state, QubitIndex qubit1, QubitIndex qubit2) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t mask1 = static_cast<std::size_t>(1) << qubit1;
  const std::size_t mask2 = static_cast<std::size_t>(1) << qubit2;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    bool b1 = (i & mask1) != 0;
    bool b2 = (i & mask2) != 0;
    if (b1 && !b2) {
      std::size_t j = (i ^ mask1) ^ mask2;
      std::swap(sv[i], sv[j]);
    }
  }
}

void apply_CRZ(QuantumState &state, QubitIndex control, QubitIndex target,
               Real theta) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t ctrl_mask = static_cast<std::size_t>(1) << control;
  const std::size_t tgt_mask = static_cast<std::size_t>(1) << target;
  const Complex phase_neg = std::exp(-constants::I * theta / 2.0);
  const Complex phase_pos = std::exp(constants::I * theta / 2.0);

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    if (i & ctrl_mask) {
      if (i & tgt_mask) {
        sv[i] *= phase_pos;
      } else {
        sv[i] *= phase_neg;
      }
    }
  }
}

// =============================================================================
// Three-Qubit Gates
// =============================================================================

void apply_Toffoli(QuantumState &state, QubitIndex control1,
                   QubitIndex control2, QubitIndex target) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t ctrl1_mask = static_cast<std::size_t>(1) << control1;
  const std::size_t ctrl2_mask = static_cast<std::size_t>(1) << control2;
  const std::size_t tgt_mask = static_cast<std::size_t>(1) << target;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    if ((i & ctrl1_mask) && (i & ctrl2_mask) && !(i & tgt_mask)) {
      std::size_t j = i ^ tgt_mask;
      std::swap(sv[i], sv[j]);
    }
  }
}

void apply_Fredkin(QuantumState &state, QubitIndex control, QubitIndex target1,
                   QubitIndex target2) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t ctrl_mask = static_cast<std::size_t>(1) << control;
  const std::size_t tgt1_mask = static_cast<std::size_t>(1) << target1;
  const std::size_t tgt2_mask = static_cast<std::size_t>(1) << target2;

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    bool ctrl = (i & ctrl_mask) != 0;
    bool t1 = (i & tgt1_mask) != 0;
    bool t2 = (i & tgt2_mask) != 0;
    if (ctrl && t1 && !t2) {
      std::size_t j = (i ^ tgt1_mask) ^ tgt2_mask;
      std::swap(sv[i], sv[j]);
    }
  }
}

void apply_MCX(QuantumState &state, const std::vector<QubitIndex> &controls,
               QubitIndex target) {
  StateVector &sv = state.state_vector();
  const std::size_t N = sv.size();
  const std::size_t tgt_mask = static_cast<std::size_t>(1) << target;

  std::size_t ctrl_mask = 0;
  for (QubitIndex c : controls) {
    ctrl_mask |= static_cast<std::size_t>(1) << c;
  }

#pragma omp parallel for
  for (std::size_t i = 0; i < N; ++i) {
    if ((i & ctrl_mask) == ctrl_mask && !(i & tgt_mask)) {
      std::size_t j = i ^ tgt_mask;
      std::swap(sv[i], sv[j]);
    }
  }
}

// =============================================================================
// Gate Dispatcher
// =============================================================================

void apply_gate(QuantumState &state, const GateInfo &gate) {
  switch (gate.type) {
  case GateType::Identity:
    break;
  case GateType::PauliX:
    apply_X(state, gate.targets[0]);
    break;
  case GateType::PauliY:
    apply_Y(state, gate.targets[0]);
    break;
  case GateType::PauliZ:
    apply_Z(state, gate.targets[0]);
    break;
  case GateType::Hadamard:
    apply_H(state, gate.targets[0]);
    break;
  case GateType::S:
    apply_S(state, gate.targets[0]);
    break;
  case GateType::T:
    apply_T(state, gate.targets[0]);
    break;
  case GateType::SDag:
    apply_Sdag(state, gate.targets[0]);
    break;
  case GateType::TDag:
    apply_Tdag(state, gate.targets[0]);
    break;
  case GateType::RX:
    apply_RX(state, gate.targets[0], gate.parameters[0]);
    break;
  case GateType::RY:
    apply_RY(state, gate.targets[0], gate.parameters[0]);
    break;
  case GateType::RZ:
    apply_RZ(state, gate.targets[0], gate.parameters[0]);
    break;
  case GateType::CNOT:
    apply_CNOT(state, gate.controls[0], gate.targets[0]);
    break;
  case GateType::CZ:
    apply_CZ(state, gate.controls[0], gate.targets[0]);
    break;
  case GateType::CY:
    apply_CY(state, gate.controls[0], gate.targets[0]);
    break;
  case GateType::SWAP:
    apply_SWAP(state, gate.targets[0], gate.targets[1]);
    break;
  case GateType::Toffoli:
    apply_Toffoli(state, gate.controls[0], gate.controls[1], gate.targets[0]);
    break;
  case GateType::Fredkin:
    apply_Fredkin(state, gate.controls[0], gate.targets[0], gate.targets[1]);
    break;
  default:
    throw std::runtime_error("Unsupported gate type");
  }
}

void apply_gate_inverse(QuantumState &state, const GateInfo &gate) {
  switch (gate.type) {
  case GateType::Identity:
  case GateType::PauliX:
  case GateType::PauliY:
  case GateType::PauliZ:
  case GateType::Hadamard:
  case GateType::CNOT:
  case GateType::CZ:
  case GateType::SWAP:
  case GateType::Toffoli:
  case GateType::Fredkin:
    apply_gate(state, gate); // Self-inverse
    break;
  case GateType::S:
    apply_Sdag(state, gate.targets[0]);
    break;
  case GateType::T:
    apply_Tdag(state, gate.targets[0]);
    break;
  case GateType::SDag:
    apply_S(state, gate.targets[0]);
    break;
  case GateType::TDag:
    apply_T(state, gate.targets[0]);
    break;
  case GateType::RX:
    apply_RX(state, gate.targets[0], -gate.parameters[0]);
    break;
  case GateType::RY:
    apply_RY(state, gate.targets[0], -gate.parameters[0]);
    break;
  case GateType::RZ:
    apply_RZ(state, gate.targets[0], -gate.parameters[0]);
    break;
  default:
    throw std::runtime_error("Unsupported gate type for inverse");
  }
}

} // namespace gates
} // namespace qmitigate
