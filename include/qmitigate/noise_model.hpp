#pragma once

#include "qmitigate/quantum_state.hpp"
#include "qmitigate/types.hpp"
#include <map>
#include <memory>
#include <random>

namespace qmitigate {

class NoiseChannelBase {
public:
  virtual ~NoiseChannelBase() = default;
  virtual void apply(QuantumState &state, QubitIndex qubit,
                     std::mt19937 &rng) = 0;
  [[nodiscard]] virtual NoiseChannel type() const = 0;
  [[nodiscard]] virtual Real probability() const = 0;
  virtual void set_scale_factor(Real scale_factor) = 0;
  [[nodiscard]] virtual Real scale_factor() const = 0;
  [[nodiscard]] virtual std::unique_ptr<NoiseChannelBase> clone() const = 0;
};

class DepolarizingChannel : public NoiseChannelBase {
public:
  explicit DepolarizingChannel(Real base_probability);
  void apply(QuantumState &state, QubitIndex qubit, std::mt19937 &rng) override;
  [[nodiscard]] NoiseChannel type() const override {
    return NoiseChannel::Depolarizing;
  }
  [[nodiscard]] Real probability() const override { return base_probability_; }
  void set_scale_factor(Real factor) override { scale_factor_ = factor; }
  [[nodiscard]] Real scale_factor() const override { return scale_factor_; }
  [[nodiscard]] std::unique_ptr<NoiseChannelBase> clone() const override;
  [[nodiscard]] Real effective_probability() const;

private:
  Real base_probability_;
  Real scale_factor_ = 1.0;
};

class BitFlipChannel : public NoiseChannelBase {
public:
  explicit BitFlipChannel(Real base_probability);
  void apply(QuantumState &state, QubitIndex qubit, std::mt19937 &rng) override;
  [[nodiscard]] NoiseChannel type() const override {
    return NoiseChannel::BitFlip;
  }
  [[nodiscard]] Real probability() const override { return base_probability_; }
  void set_scale_factor(Real factor) override { scale_factor_ = factor; }
  [[nodiscard]] Real scale_factor() const override { return scale_factor_; }
  [[nodiscard]] std::unique_ptr<NoiseChannelBase> clone() const override;

private:
  Real base_probability_;
  Real scale_factor_ = 1.0;
};

class PhaseFlipChannel : public NoiseChannelBase {
public:
  explicit PhaseFlipChannel(Real base_probability);
  void apply(QuantumState &state, QubitIndex qubit, std::mt19937 &rng) override;
  [[nodiscard]] NoiseChannel type() const override {
    return NoiseChannel::PhaseFlip;
  }
  [[nodiscard]] Real probability() const override { return base_probability_; }
  void set_scale_factor(Real factor) override { scale_factor_ = factor; }
  [[nodiscard]] Real scale_factor() const override { return scale_factor_; }
  [[nodiscard]] std::unique_ptr<NoiseChannelBase> clone() const override;

private:
  Real base_probability_;
  Real scale_factor_ = 1.0;
};

class NoiseModel {
public:
  NoiseModel() = default;
  explicit NoiseModel(std::unique_ptr<NoiseChannelBase> channel);
  NoiseModel(const NoiseModel &other);
  NoiseModel(NoiseModel &&other) noexcept = default;
  NoiseModel &operator=(const NoiseModel &other);
  NoiseModel &operator=(NoiseModel &&other) noexcept = default;

  void add_default_noise(std::unique_ptr<NoiseChannelBase> channel);
  void apply_noise(QuantumState &state, const GateInfo &gate,
                   std::mt19937 &rng);
  void set_scale_factor(Real scale_factor);
  [[nodiscard]] Real scale_factor() const { return scale_factor_; }
  [[nodiscard]] bool has_noise() const { return !default_channels_.empty(); }
  void seed(unsigned int seed_value) { rng_.seed(seed_value); }

private:
  std::vector<std::unique_ptr<NoiseChannelBase>> default_channels_;
  Real scale_factor_ = 1.0;
  std::mt19937 rng_{std::random_device{}()};
};

[[nodiscard]] NoiseModel
create_depolarizing_noise_model(Real error_probability);

} // namespace qmitigate
