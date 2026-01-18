/**
 * @file benchmark_engine.cpp
 * @brief Performance benchmarks for QMitigate engine
 *
 * This demonstrates the HPC optimization advantage of C++ over Python.
 */

#include "qmitigate/simulator.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace qmitigate;

struct BenchmarkResult {
  int num_qubits;
  int circuit_depth;
  double mean_time_ms;
  double std_time_ms;
  double gates_per_second;
  double state_size_mb;
};

BenchmarkResult benchmark_circuit(int num_qubits, int circuit_depth,
                                  int num_runs = 5) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> gate_dist(0, 3);
  std::uniform_int_distribution<int> qubit_dist(0, num_qubits - 1);
  std::uniform_real_distribution<double> angle_dist(0, 3.14159);

  // Build a random circuit
  Circuit circuit(static_cast<QubitIndex>(num_qubits));
  for (int i = 0; i < circuit_depth; ++i) {
    int gate_type = gate_dist(rng);
    int qubit = qubit_dist(rng);

    switch (gate_type) {
    case 0:
      circuit.h(static_cast<QubitIndex>(qubit));
      break;
    case 1:
      circuit.x(static_cast<QubitIndex>(qubit));
      break;
    case 2:
      circuit.rx(static_cast<QubitIndex>(qubit), angle_dist(rng));
      break;
    case 3:
      if (num_qubits > 1) {
        int target = (qubit + 1) % num_qubits;
        circuit.cnot(static_cast<QubitIndex>(qubit),
                     static_cast<QubitIndex>(target));
      }
      break;
    }
  }

  Simulator sim;

  // Warm-up run
  sim.run(circuit);

  // Timed runs
  std::vector<double> times;
  for (int i = 0; i < num_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto state = sim.run(circuit);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double, std::milli>(end - start);
    times.push_back(duration.count());
  }

  // Calculate statistics
  double sum = 0.0;
  for (double t : times)
    sum += t;
  double mean = sum / times.size();

  double var_sum = 0.0;
  for (double t : times)
    var_sum += (t - mean) * (t - mean);
  double std_dev = std::sqrt(var_sum / times.size());

  BenchmarkResult result;
  result.num_qubits = num_qubits;
  result.circuit_depth = circuit_depth;
  result.mean_time_ms = mean;
  result.std_time_ms = std_dev;
  result.gates_per_second = circuit_depth / (mean / 1000.0);
  result.state_size_mb =
      static_cast<double>(1ULL << num_qubits) * 16.0 / (1024.0 * 1024.0);

  return result;
}

void print_result(const BenchmarkResult &r) {
  std::cout << std::setw(7) << r.num_qubits << std::setw(12) << r.circuit_depth
            << std::setw(14) << std::fixed << std::setprecision(2)
            << r.state_size_mb << std::setw(14) << std::fixed
            << std::setprecision(3) << r.mean_time_ms << std::setw(12)
            << std::fixed << std::setprecision(3) << r.std_time_ms
            << std::setw(16) << std::fixed << std::setprecision(0)
            << r.gates_per_second << std::endl;
}

int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════════"
               "════════════════╗\n";
  std::cout << "║             QMitigate High-Performance Quantum Simulator "
               "Benchmark            ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════════"
               "════════════════╝\n\n";

  std::cout << "System: C++20 with OpenMP parallelization\n";
  std::cout << "Optimization: O(2^N) sparse gate application (vs O(4^N) matrix "
               "multiply)\n\n";

  // Header
  std::cout << std::setw(7) << "Qubits" << std::setw(12) << "Depth"
            << std::setw(14) << "State (MB)" << std::setw(14) << "Time (ms)"
            << std::setw(12) << "Std (ms)" << std::setw(16) << "Gates/sec"
            << std::endl;
  std::cout << std::string(75, '-') << std::endl;

  // Scaling benchmark
  std::vector<int> qubit_counts = {5, 8, 10, 12, 14, 16, 18, 20};
  int depth = 50;

  for (int n : qubit_counts) {
    try {
      auto result = benchmark_circuit(n, depth);
      print_result(result);
    } catch (const std::exception &e) {
      std::cout << "Qubits " << n << ": Error - " << e.what() << std::endl;
    }
  }

  std::cout << std::string(75, '-') << std::endl;

  // Deep circuit benchmark
  std::cout << "\nDeep Circuit Benchmark (15 qubits, varying depth):\n";
  std::cout << std::string(75, '-') << std::endl;

  std::vector<int> depths = {10, 50, 100, 200, 500};
  for (int d : depths) {
    auto result = benchmark_circuit(15, d);
    print_result(result);
  }

  std::cout << std::string(75, '-') << std::endl;
  std::cout << "\nBenchmark complete.\n\n";

  return 0;
}
