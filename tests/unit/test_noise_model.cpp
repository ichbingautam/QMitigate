/**
 * @file test_noise_model.cpp
 * @brief Unit tests for noise channels and models
 */

#include "qmitigate/noise_model.hpp"
#include "qmitigate/quantum_state.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace qmitigate;

class NoiseModelTest : public ::testing::Test {
protected:
  static constexpr double TOLERANCE = 1e-6;
  std::mt19937 rng{42};
};

TEST_F(NoiseModelTest, DepolarizingChannel_ZeroProbability) {
  QuantumState state(1);
  DepolarizingChannel channel(0.0);

  Complex before = state.amplitude(0);
  channel.apply(state, 0, rng);
  Complex after = state.amplitude(0);

  // Zero probability means no change
  EXPECT_NEAR(std::abs(before - after), 0.0, TOLERANCE);
}

TEST_F(NoiseModelTest, DepolarizingChannel_FullProbability) {
  // With probability 1.0, every application adds an error
  // But for |0⟩ state, Z|0⟩ = |0⟩ (no visible change)
  // Only X (bit flip) and Y errors are detectable by amplitude change
  // Since X, Y, Z are equally probable (1/3 each), expect ~66% changes
  DepolarizingChannel channel(1.0);

  int changes = 0;
  for (int i = 0; i < 100; ++i) {
    QuantumState state(1);
    Complex before = state.amplitude(0);
    channel.apply(state, 0, rng);
    Complex after = state.amplitude(0);

    if (std::abs(before - after) > TOLERANCE) {
      ++changes;
    }
  }

  // Expect roughly 2/3 of applications to show visible change (X or Y error)
  // Allow some statistical variation (50-85 out of 100)
  EXPECT_GT(changes, 50);
  EXPECT_LT(changes, 85);
}

TEST_F(NoiseModelTest, DepolarizingChannel_ScaleFactor) {
  DepolarizingChannel channel(0.1);

  EXPECT_NEAR(channel.probability(), 0.1, TOLERANCE);
  EXPECT_NEAR(channel.scale_factor(), 1.0, TOLERANCE);
  EXPECT_NEAR(channel.effective_probability(), 0.1, TOLERANCE);

  channel.set_scale_factor(3.0);
  EXPECT_NEAR(channel.effective_probability(), 0.3, TOLERANCE);

  channel.set_scale_factor(20.0); // Would be 2.0, capped at 1.0
  EXPECT_NEAR(channel.effective_probability(), 1.0, TOLERANCE);
}

TEST_F(NoiseModelTest, BitFlipChannel) {
  BitFlipChannel channel(1.0); // Always flip

  QuantumState state(1); // |0⟩
  channel.apply(state, 0, rng);

  // Should now be |1⟩
  EXPECT_NEAR(state.probability(1), 1.0, TOLERANCE);
}

TEST_F(NoiseModelTest, PhaseFlipChannel) {
  PhaseFlipChannel channel(1.0); // Always flip phase

  QuantumState state(1);
  state.initialize_basis_state(1); // |1⟩

  Complex before = state.amplitude(1);
  channel.apply(state, 0, rng);
  Complex after = state.amplitude(1);

  // Z|1⟩ = -|1⟩
  EXPECT_NEAR((after / before).real(), -1.0, TOLERANCE);
}

TEST_F(NoiseModelTest, NoiseModel_NoNoise) {
  NoiseModel model;

  EXPECT_FALSE(model.has_noise());
}

TEST_F(NoiseModelTest, NoiseModel_WithDepolarizing) {
  NoiseModel model = create_depolarizing_noise_model(0.01);

  EXPECT_TRUE(model.has_noise());
  EXPECT_NEAR(model.scale_factor(), 1.0, TOLERANCE);
}

TEST_F(NoiseModelTest, NoiseModel_ScaleFactorPropagates) {
  auto channel = std::make_unique<DepolarizingChannel>(0.1);
  NoiseModel model(std::move(channel));

  model.set_scale_factor(5.0);
  EXPECT_NEAR(model.scale_factor(), 5.0, TOLERANCE);
}

TEST_F(NoiseModelTest, NoiseModel_Copy) {
  NoiseModel original = create_depolarizing_noise_model(0.05);
  original.set_scale_factor(3.0);

  NoiseModel copy = original;

  EXPECT_TRUE(copy.has_noise());
  EXPECT_NEAR(copy.scale_factor(), 3.0, TOLERANCE);
}

TEST_F(NoiseModelTest, DepolarizingChannel_Clone) {
  DepolarizingChannel channel(0.1);
  channel.set_scale_factor(2.0);

  auto clone = channel.clone();

  EXPECT_NEAR(clone->probability(), 0.1, TOLERANCE);
  EXPECT_NEAR(clone->scale_factor(), 2.0, TOLERANCE);
  EXPECT_EQ(clone->type(), NoiseChannel::Depolarizing);
}
