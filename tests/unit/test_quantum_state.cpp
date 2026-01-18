/**
 * @file test_quantum_state.cpp
 * @brief Unit tests for QuantumState class
 */

#include "qmitigate/quantum_state.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace qmitigate;

class QuantumStateTest : public ::testing::Test {
protected:
  static constexpr double TOLERANCE = 1e-10;
};

TEST_F(QuantumStateTest, InitializesToZeroState) {
  QuantumState state(3);

  EXPECT_EQ(state.num_qubits(), 3u);
  EXPECT_EQ(state.dimension(), 8u);

  // |000⟩ should have amplitude 1
  EXPECT_NEAR(std::abs(state.amplitude(0)), 1.0, TOLERANCE);

  // All other amplitudes should be 0
  for (std::size_t i = 1; i < state.dimension(); ++i) {
    EXPECT_NEAR(std::abs(state.amplitude(i)), 0.0, TOLERANCE);
  }
}

TEST_F(QuantumStateTest, IsNormalized) {
  QuantumState state(5);
  EXPECT_TRUE(state.is_normalized());
  EXPECT_NEAR(state.norm(), 1.0, TOLERANCE);
}

TEST_F(QuantumStateTest, InitializeBasisState) {
  QuantumState state(3);
  state.initialize_basis_state(5); // |101⟩

  EXPECT_NEAR(std::abs(state.amplitude(5)), 1.0, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(0)), 0.0, TOLERANCE);
  EXPECT_TRUE(state.is_normalized());
}

TEST_F(QuantumStateTest, InitializeSuperposition) {
  QuantumState state(3);
  state.initialize_superposition();

  double expected_amp = 1.0 / std::sqrt(8.0);
  for (std::size_t i = 0; i < state.dimension(); ++i) {
    EXPECT_NEAR(std::abs(state.amplitude(i)), expected_amp, TOLERANCE);
  }
  EXPECT_TRUE(state.is_normalized());
}

TEST_F(QuantumStateTest, Reset) {
  QuantumState state(3);
  state.initialize_basis_state(7);
  state.reset();

  EXPECT_NEAR(std::abs(state.amplitude(0)), 1.0, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(7)), 0.0, TOLERANCE);
}

TEST_F(QuantumStateTest, Fidelity) {
  QuantumState state1(2);
  QuantumState state2(2);

  EXPECT_NEAR(state1.fidelity(state2), 1.0, TOLERANCE);

  state2.initialize_basis_state(3);
  EXPECT_NEAR(state1.fidelity(state2), 0.0, TOLERANCE);
}

TEST_F(QuantumStateTest, InnerProduct) {
  QuantumState state1(2);
  QuantumState state2(2);

  Complex ip = state1.inner_product(state2);
  EXPECT_NEAR(std::abs(ip), 1.0, TOLERANCE);

  state2.initialize_basis_state(1);
  ip = state1.inner_product(state2);
  EXPECT_NEAR(std::abs(ip), 0.0, TOLERANCE);
}

TEST_F(QuantumStateTest, ProbabilityZero) {
  QuantumState state(2);

  // |00⟩ - qubit 0 is |0⟩ with probability 1
  EXPECT_NEAR(state.probability_zero(0), 1.0, TOLERANCE);
  EXPECT_NEAR(state.probability_zero(1), 1.0, TOLERANCE);
}

TEST_F(QuantumStateTest, ExpectationZ) {
  QuantumState state(2);

  // |00⟩ - ⟨Z⟩ = +1 for both qubits
  EXPECT_NEAR(state.expectation_Z(0), 1.0, TOLERANCE);
  EXPECT_NEAR(state.expectation_Z(1), 1.0, TOLERANCE);

  state.initialize_basis_state(3); // |11⟩
  EXPECT_NEAR(state.expectation_Z(0), -1.0, TOLERANCE);
  EXPECT_NEAR(state.expectation_Z(1), -1.0, TOLERANCE);
}

TEST_F(QuantumStateTest, BellStateCreation) {
  QuantumState bell = create_bell_state();

  EXPECT_EQ(bell.num_qubits(), 2u);
  EXPECT_TRUE(bell.is_normalized());

  // |00⟩ and |11⟩ should have equal amplitude
  EXPECT_NEAR(std::abs(bell.amplitude(0)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(bell.amplitude(3)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(bell.amplitude(1)), 0.0, TOLERANCE);
  EXPECT_NEAR(std::abs(bell.amplitude(2)), 0.0, TOLERANCE);
}

TEST_F(QuantumStateTest, GHZStateCreation) {
  QuantumState ghz = create_ghz_state(4);

  EXPECT_EQ(ghz.num_qubits(), 4u);
  EXPECT_TRUE(ghz.is_normalized());

  // |0000⟩ and |1111⟩ should have equal amplitude
  EXPECT_NEAR(std::abs(ghz.amplitude(0)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(ghz.amplitude(15)), constants::INV_SQRT2, TOLERANCE);
}

TEST_F(QuantumStateTest, WStateCreation) {
  QuantumState w = create_w_state(3);

  EXPECT_EQ(w.num_qubits(), 3u);
  EXPECT_TRUE(w.is_normalized());

  double expected = 1.0 / std::sqrt(3.0);
  EXPECT_NEAR(std::abs(w.amplitude(1)), expected, TOLERANCE); // |001⟩
  EXPECT_NEAR(std::abs(w.amplitude(2)), expected, TOLERANCE); // |010⟩
  EXPECT_NEAR(std::abs(w.amplitude(4)), expected, TOLERANCE); // |100⟩
}

TEST_F(QuantumStateTest, MaxQubitsCheck) {
  // 50 qubits would require 2^50 * 16 bytes = ~18 EB of RAM
  // This will fail either with invalid_argument (if check happens first)
  // or bad_alloc (if allocation attempt fails first)
  EXPECT_ANY_THROW(QuantumState(50));
}

TEST_F(QuantumStateTest, ZeroQubitsCheck) {
  EXPECT_THROW(QuantumState(0), std::invalid_argument);
}
