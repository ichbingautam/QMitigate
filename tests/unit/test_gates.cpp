/**
 * @file test_gates.cpp
 * @brief Unit tests for quantum gate operations
 */

#include "qmitigate/gates.hpp"
#include "qmitigate/quantum_state.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace qmitigate;

class GatesTest : public ::testing::Test {
protected:
  static constexpr double TOLERANCE = 1e-10;
};

// =============================================================================
// Single-Qubit Gates
// =============================================================================

TEST_F(GatesTest, PauliX_FlipsBit) {
  QuantumState state(2); // |00⟩
  gates::apply_X(state, 0);

  // Should now be |01⟩
  EXPECT_NEAR(std::abs(state.amplitude(1)), 1.0, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(0)), 0.0, TOLERANCE);
  EXPECT_TRUE(state.is_normalized());
}

TEST_F(GatesTest, PauliX_SelfInverse) {
  QuantumState state(2);
  gates::apply_X(state, 0);
  gates::apply_X(state, 0);

  // Should be back to |00⟩
  EXPECT_NEAR(std::abs(state.amplitude(0)), 1.0, TOLERANCE);
}

TEST_F(GatesTest, PauliZ_PhaseFlip) {
  QuantumState state(1);
  state.initialize_basis_state(1); // |1⟩

  Complex before = state.amplitude(1);
  gates::apply_Z(state, 0);
  Complex after = state.amplitude(1);

  // Z|1⟩ = -|1⟩
  EXPECT_NEAR((after / before).real(), -1.0, TOLERANCE);
  EXPECT_NEAR((after / before).imag(), 0.0, TOLERANCE);
}

TEST_F(GatesTest, Hadamard_CreatesSuperposition) {
  QuantumState state(1); // |0⟩
  gates::apply_H(state, 0);

  // H|0⟩ = (|0⟩ + |1⟩)/√2
  EXPECT_NEAR(std::abs(state.amplitude(0)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(1)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_TRUE(state.is_normalized());
}

TEST_F(GatesTest, Hadamard_SelfInverse) {
  QuantumState state(1);
  gates::apply_H(state, 0);
  gates::apply_H(state, 0);

  // H² = I
  EXPECT_NEAR(std::abs(state.amplitude(0)), 1.0, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(1)), 0.0, TOLERANCE);
}

TEST_F(GatesTest, S_Gate) {
  QuantumState state(1);
  state.initialize_basis_state(1); // |1⟩
  gates::apply_S(state, 0);

  // S|1⟩ = i|1⟩
  Complex amp = state.amplitude(1);
  EXPECT_NEAR(amp.imag(), 1.0, TOLERANCE);
  EXPECT_NEAR(amp.real(), 0.0, TOLERANCE);
}

TEST_F(GatesTest, T_Gate) {
  QuantumState state(1);
  state.initialize_basis_state(1); // |1⟩
  gates::apply_T(state, 0);

  // T|1⟩ = e^(iπ/4)|1⟩
  Complex expected = std::exp(constants::I * constants::PI / 4.0);
  Complex amp = state.amplitude(1);
  EXPECT_NEAR(amp.real(), expected.real(), TOLERANCE);
  EXPECT_NEAR(amp.imag(), expected.imag(), TOLERANCE);
}

// =============================================================================
// Rotation Gates
// =============================================================================

TEST_F(GatesTest, RX_HalfPi_EqualsHadamardRotation) {
  QuantumState state1(1);
  QuantumState state2(1);

  // RX(π) should flip the qubit (like X but with a phase)
  gates::apply_RX(state1, 0, constants::PI);

  // |0⟩ → -i|1⟩
  EXPECT_NEAR(std::abs(state1.amplitude(0)), 0.0, TOLERANCE);
  EXPECT_NEAR(std::abs(state1.amplitude(1)), 1.0, TOLERANCE);
}

TEST_F(GatesTest, RY_Pi_FlipsQubit) {
  QuantumState state(1);
  gates::apply_RY(state, 0, constants::PI);

  // RY(π)|0⟩ = |1⟩
  EXPECT_NEAR(std::abs(state.amplitude(0)), 0.0, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(1)), 1.0, TOLERANCE);
}

TEST_F(GatesTest, RZ_FullRotation_Identity) {
  QuantumState state(1);
  gates::apply_H(state, 0); // Create superposition
  QuantumState original = state;

  gates::apply_RZ(state, 0, 2 * constants::PI);

  // RZ(2π) should give same probabilities (global phase)
  EXPECT_NEAR(state.probability(0), original.probability(0), TOLERANCE);
  EXPECT_NEAR(state.probability(1), original.probability(1), TOLERANCE);
}

// =============================================================================
// Two-Qubit Gates
// =============================================================================

TEST_F(GatesTest, CNOT_CreatesEntanglement) {
  QuantumState state(2);
  gates::apply_H(state, 0); // |0⟩ → |+⟩
  gates::apply_CNOT(state, 0, 1);

  // Should create Bell state (|00⟩ + |11⟩)/√2
  EXPECT_NEAR(std::abs(state.amplitude(0)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(3)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(1)), 0.0, TOLERANCE);
  EXPECT_NEAR(std::abs(state.amplitude(2)), 0.0, TOLERANCE);
}

TEST_F(GatesTest, CNOT_ControlOff_NoAction) {
  QuantumState state(2); // |00⟩
  gates::apply_CNOT(state, 0, 1);

  // Control is 0, so nothing happens
  EXPECT_NEAR(std::abs(state.amplitude(0)), 1.0, TOLERANCE);
}

TEST_F(GatesTest, CNOT_ControlOn_Flips) {
  QuantumState state(2);
  gates::apply_X(state, 0); // |01⟩ (qubit 0 is rightmost)
  gates::apply_CNOT(state, 0, 1);

  // Control is 1, so target flips: |01⟩ → |11⟩
  EXPECT_NEAR(std::abs(state.amplitude(3)), 1.0, TOLERANCE);
}

TEST_F(GatesTest, CZ_Gate) {
  QuantumState state(2);
  state.initialize_basis_state(3); // |11⟩

  Complex before = state.amplitude(3);
  gates::apply_CZ(state, 0, 1);
  Complex after = state.amplitude(3);

  // CZ|11⟩ = -|11⟩
  EXPECT_NEAR((after / before).real(), -1.0, TOLERANCE);
}

TEST_F(GatesTest, SWAP_Gate) {
  QuantumState state(2);
  state.initialize_basis_state(1); // |01⟩ (qubit 0 = 1, qubit 1 = 0)
  gates::apply_SWAP(state, 0, 1);

  // Should become |10⟩
  EXPECT_NEAR(std::abs(state.amplitude(2)), 1.0, TOLERANCE);
}

// =============================================================================
// Three-Qubit Gates
// =============================================================================

TEST_F(GatesTest, Toffoli_BothControlsOn) {
  QuantumState state(3);
  gates::apply_X(state, 0); // Set qubit 0
  gates::apply_X(state, 1); // Set qubit 1
  // State is now |011⟩

  gates::apply_Toffoli(state, 0, 1, 2);

  // Both controls on, so target flips: |011⟩ → |111⟩
  EXPECT_NEAR(std::abs(state.amplitude(7)), 1.0, TOLERANCE);
}

TEST_F(GatesTest, Toffoli_OneControlOff) {
  QuantumState state(3);
  gates::apply_X(state, 0); // Set only qubit 0: |001⟩

  gates::apply_Toffoli(state, 0, 1, 2);

  // Not both controls on, so no change
  EXPECT_NEAR(std::abs(state.amplitude(1)), 1.0, TOLERANCE);
}

// =============================================================================
// Gate Inverses
// =============================================================================

TEST_F(GatesTest, GateInverse_S) {
  QuantumState state(1);
  state.initialize_basis_state(1);

  gates::apply_S(state, 0);
  gates::apply_Sdag(state, 0);

  // S†S = I
  EXPECT_NEAR(state.amplitude(1).real(), 1.0, TOLERANCE);
  EXPECT_NEAR(state.amplitude(1).imag(), 0.0, TOLERANCE);
}

TEST_F(GatesTest, GateInverse_RX) {
  QuantumState state(1);
  gates::apply_H(state, 0);
  QuantumState original = state;

  gates::apply_RX(state, 0, 0.5);
  gates::apply_RX(state, 0, -0.5);

  // RX(θ)RX(-θ) = I
  EXPECT_NEAR(state.fidelity(original), 1.0, TOLERANCE);
}
