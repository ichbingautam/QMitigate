/**
 * @file test_simulator.cpp
 * @brief Unit tests for Simulator and digital folding
 */

#include "qmitigate/simulator.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace qmitigate;

class SimulatorTest : public ::testing::Test {
protected:
  static constexpr double TOLERANCE = 1e-10;
};

TEST_F(SimulatorTest, RunEmptyCircuit) {
  Circuit circuit(2);
  Simulator sim;

  QuantumState result = sim.run(circuit);

  // Empty circuit should leave |00⟩
  EXPECT_NEAR(std::abs(result.amplitude(0)), 1.0, TOLERANCE);
}

TEST_F(SimulatorTest, RunBellCircuit) {
  Circuit circuit(2);
  circuit.h(0);
  circuit.cnot(0, 1);

  Simulator sim;
  QuantumState result = sim.run(circuit);

  // Bell state (|00⟩ + |11⟩)/√2
  EXPECT_NEAR(std::abs(result.amplitude(0)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(result.amplitude(3)), constants::INV_SQRT2, TOLERANCE);
  EXPECT_NEAR(std::abs(result.amplitude(1)), 0.0, TOLERANCE);
  EXPECT_NEAR(std::abs(result.amplitude(2)), 0.0, TOLERANCE);
}

TEST_F(SimulatorTest, CircuitDepth) {
  Circuit circuit(3);
  circuit.h(0);
  circuit.x(1);
  circuit.cnot(0, 2);

  EXPECT_EQ(circuit.depth(), 3u);
}

TEST_F(SimulatorTest, CircuitBuilderMethods) {
  Circuit circuit(2);
  circuit.h(0);
  circuit.x(1);
  circuit.y(0);
  circuit.z(1);
  circuit.rx(0, 0.5);
  circuit.ry(1, 0.5);
  circuit.rz(0, 0.5);
  circuit.cnot(0, 1);

  EXPECT_EQ(circuit.depth(), 8u);
  EXPECT_EQ(circuit.num_qubits, 2u);
}

TEST_F(SimulatorTest, RunWithNoise) {
  Circuit circuit(2);
  circuit.h(0);
  circuit.cnot(0, 1);

  NoiseModel noise = create_depolarizing_noise_model(0.01);
  Simulator sim(noise);

  QuantumState result = sim.run(circuit);

  // With noise, probabilities should still sum to 1
  auto probs = result.get_probabilities();
  Real total = 0.0;
  for (auto p : probs)
    total += p;
  EXPECT_NEAR(total, 1.0, 1e-6);
}

TEST_F(SimulatorTest, NoiseScaleChange) {
  NoiseModel noise = create_depolarizing_noise_model(0.01);
  Simulator sim(noise);

  EXPECT_NEAR(sim.noise_scale(), 1.0, TOLERANCE);

  sim.set_noise_scale(3.0);
  EXPECT_NEAR(sim.noise_scale(), 3.0, TOLERANCE);
}

// =============================================================================
// Digital Folding Tests
// =============================================================================

TEST_F(SimulatorTest, FoldCircuit_Scale1_NoChange) {
  Circuit original(2);
  original.h(0);
  original.cnot(0, 1);

  Circuit folded = fold_circuit_global(original, 1);

  EXPECT_EQ(folded.depth(), original.depth());
}

TEST_F(SimulatorTest, FoldCircuit_Scale3_TriplesEffectiveDepth) {
  Circuit original(2);
  original.h(0);
  original.cnot(0, 1);

  // Scale 3: U → U U† U
  Circuit folded = fold_circuit_global(original, 3);

  // Original: 2 gates
  // Folded: 2 + 2 (inverse) + 2 (original) = 6 gates
  EXPECT_EQ(folded.depth(), 3 * original.depth());
}

TEST_F(SimulatorTest, FoldCircuit_Scale5) {
  Circuit original(2);
  original.h(0);

  // Scale 5: U → U U† U U† U
  Circuit folded = fold_circuit_global(original, 5);

  // 2 folds: original + 2*(inverse + original) = 1 + 2*2 = 5
  EXPECT_EQ(folded.depth(), 5 * original.depth());
}

TEST_F(SimulatorTest, FoldCircuit_InvalidEvenScale) {
  Circuit circuit(1);
  circuit.h(0);

  EXPECT_THROW(fold_circuit_global(circuit, 2), std::invalid_argument);
  EXPECT_THROW(fold_circuit_global(circuit, 4), std::invalid_argument);
}

TEST_F(SimulatorTest, FoldedCircuit_SameLogicalResult_NoNoise) {
  Circuit original(2);
  original.h(0);
  original.cnot(0, 1);

  Simulator sim; // No noise

  QuantumState result1 = sim.run(original);

  Circuit folded = fold_circuit_global(original, 3);
  QuantumState result2 = sim.run(folded);

  // Without noise, folded circuit should give same result
  // (U U† U = U)
  EXPECT_NEAR(result1.fidelity(result2), 1.0, 1e-6);
}

TEST_F(SimulatorTest, ExpectationZ_BellState) {
  Circuit circuit(2);
  circuit.h(0);
  circuit.cnot(0, 1);

  Simulator sim;

  // For Bell state, ⟨Z₀⟩ = 0 (equal superposition of |0⟩ and |1⟩)
  Real exp_z = sim.expectation_Z(circuit, 0, 100);
  EXPECT_NEAR(exp_z, 0.0, 0.2); // Statistical tolerance
}

TEST_F(SimulatorTest, ExpectationZ_ZeroState) {
  Circuit circuit(1); // Empty circuit, |0⟩

  Simulator sim;

  // ⟨Z⟩ = 1 for |0⟩
  Real exp_z = sim.expectation_Z(circuit, 0, 100);
  EXPECT_NEAR(exp_z, 1.0, 0.01);
}
