/**
 * @file noise_model.cpp
 * @brief Implementation of noise channels for NISQ error simulation
 */

#include "qmitigate/noise_model.hpp"
#include "qmitigate/gates.hpp"
#include <algorithm>

namespace qmitigate {

// =============================================================================
// Depolarizing Channel
// =============================================================================

DepolarizingChannel::DepolarizingChannel(Real base_probability)
    : base_probability_(std::clamp(base_probability, 0.0, 1.0)) {}

void DepolarizingChannel::apply(QuantumState &state, QubitIndex qubit,
                                std::mt19937 &rng) {
  Real p_eff = effective_probability();
  if (p_eff < constants::TOLERANCE)
    return;

  std::uniform_real_distribution<Real> dist(0.0, 1.0);
  Real r = dist(rng);

  if (r < p_eff) {
    // Apply a random Pauli error
    Real error_choice = dist(rng);
    if (error_choice < 1.0 / 3.0) {
      gates::apply_X(state, qubit);
    } else if (error_choice < 2.0 / 3.0) {
      gates::apply_Y(state, qubit);
    } else {
      gates::apply_Z(state, qubit);
    }
  }
}

std::unique_ptr<NoiseChannelBase> DepolarizingChannel::clone() const {
  auto copy = std::make_unique<DepolarizingChannel>(base_probability_);
  copy->set_scale_factor(scale_factor_);
  return copy;
}

Real DepolarizingChannel::effective_probability() const {
  return std::min(base_probability_ * scale_factor_, 1.0);
}

// =============================================================================
// Bit Flip Channel
// =============================================================================

BitFlipChannel::BitFlipChannel(Real base_probability)
    : base_probability_(std::clamp(base_probability, 0.0, 1.0)) {}

void BitFlipChannel::apply(QuantumState &state, QubitIndex qubit,
                           std::mt19937 &rng) {
  Real p_eff = std::min(base_probability_ * scale_factor_, 1.0);
  if (p_eff < constants::TOLERANCE)
    return;

  std::uniform_real_distribution<Real> dist(0.0, 1.0);
  if (dist(rng) < p_eff) {
    gates::apply_X(state, qubit);
  }
}

std::unique_ptr<NoiseChannelBase> BitFlipChannel::clone() const {
  auto copy = std::make_unique<BitFlipChannel>(base_probability_);
  copy->set_scale_factor(scale_factor_);
  return copy;
}

// =============================================================================
// Phase Flip Channel
// =============================================================================

PhaseFlipChannel::PhaseFlipChannel(Real base_probability)
    : base_probability_(std::clamp(base_probability, 0.0, 1.0)) {}

void PhaseFlipChannel::apply(QuantumState &state, QubitIndex qubit,
                             std::mt19937 &rng) {
  Real p_eff = std::min(base_probability_ * scale_factor_, 1.0);
  if (p_eff < constants::TOLERANCE)
    return;

  std::uniform_real_distribution<Real> dist(0.0, 1.0);
  if (dist(rng) < p_eff) {
    gates::apply_Z(state, qubit);
  }
}

std::unique_ptr<NoiseChannelBase> PhaseFlipChannel::clone() const {
  auto copy = std::make_unique<PhaseFlipChannel>(base_probability_);
  copy->set_scale_factor(scale_factor_);
  return copy;
}

// =============================================================================
// Noise Model
// =============================================================================

NoiseModel::NoiseModel(std::unique_ptr<NoiseChannelBase> channel) {
  default_channels_.push_back(std::move(channel));
}

NoiseModel::NoiseModel(const NoiseModel &other)
    : scale_factor_(other.scale_factor_) {
  for (const auto &ch : other.default_channels_) {
    default_channels_.push_back(ch->clone());
  }
}

NoiseModel &NoiseModel::operator=(const NoiseModel &other) {
  if (this != &other) {
    default_channels_.clear();
    scale_factor_ = other.scale_factor_;
    for (const auto &ch : other.default_channels_) {
      default_channels_.push_back(ch->clone());
    }
  }
  return *this;
}

void NoiseModel::add_default_noise(std::unique_ptr<NoiseChannelBase> channel) {
  default_channels_.push_back(std::move(channel));
}

void NoiseModel::apply_noise(QuantumState &state, const GateInfo &gate,
                             std::mt19937 &rng) {
  // Apply default noise to all target qubits
  for (QubitIndex q : gate.targets) {
    for (auto &channel : default_channels_) {
      channel->apply(state, q, rng);
    }
  }
  // Also apply to control qubits (they experience decoherence too)
  for (QubitIndex q : gate.controls) {
    for (auto &channel : default_channels_) {
      channel->apply(state, q, rng);
    }
  }
}

void NoiseModel::set_scale_factor(Real scale_factor) {
  scale_factor_ = scale_factor;
  for (auto &channel : default_channels_) {
    channel->set_scale_factor(scale_factor);
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

NoiseModel create_depolarizing_noise_model(Real error_probability) {
  return NoiseModel(std::make_unique<DepolarizingChannel>(error_probability));
}

} // namespace qmitigate
