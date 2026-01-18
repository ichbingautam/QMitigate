/**
 * @file python_bindings.cpp
 * @brief PyBind11 bindings exposing QMitigate to Python
 */

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qmitigate/gates.hpp"
#include "qmitigate/noise_model.hpp"
#include "qmitigate/quantum_state.hpp"
#include "qmitigate/simulator.hpp"
#include "qmitigate/types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qmitigate_cpp, m) {
  m.doc() =
      "QMitigate: High-performance quantum simulator with error mitigation";

  // =========================================================================
  // Enums
  // =========================================================================
  py::enum_<qmitigate::GateType>(m, "GateType")
      .value("Identity", qmitigate::GateType::Identity)
      .value("PauliX", qmitigate::GateType::PauliX)
      .value("PauliY", qmitigate::GateType::PauliY)
      .value("PauliZ", qmitigate::GateType::PauliZ)
      .value("Hadamard", qmitigate::GateType::Hadamard)
      .value("S", qmitigate::GateType::S)
      .value("T", qmitigate::GateType::T)
      .value("RX", qmitigate::GateType::RX)
      .value("RY", qmitigate::GateType::RY)
      .value("RZ", qmitigate::GateType::RZ)
      .value("CNOT", qmitigate::GateType::CNOT)
      .value("CZ", qmitigate::GateType::CZ)
      .value("SWAP", qmitigate::GateType::SWAP)
      .value("Toffoli", qmitigate::GateType::Toffoli);

  py::enum_<qmitigate::NoiseChannel>(m, "NoiseChannel")
      .value("None", qmitigate::NoiseChannel::None)
      .value("Depolarizing", qmitigate::NoiseChannel::Depolarizing)
      .value("BitFlip", qmitigate::NoiseChannel::BitFlip)
      .value("PhaseFlip", qmitigate::NoiseChannel::PhaseFlip);

  // =========================================================================
  // QuantumState
  // =========================================================================
  py::class_<qmitigate::QuantumState>(m, "QuantumState")
      .def(py::init<qmitigate::QubitIndex>(), py::arg("num_qubits"))
      .def(py::init<qmitigate::StateVector>(), py::arg("state_vector"))
      .def("num_qubits", &qmitigate::QuantumState::num_qubits)
      .def("dimension", &qmitigate::QuantumState::dimension)
      .def("amplitude", &qmitigate::QuantumState::amplitude, py::arg("index"))
      .def("probability", &qmitigate::QuantumState::probability,
           py::arg("index"))
      .def("norm", &qmitigate::QuantumState::norm)
      .def("normalize", &qmitigate::QuantumState::normalize)
      .def("is_normalized", &qmitigate::QuantumState::is_normalized)
      .def("fidelity", &qmitigate::QuantumState::fidelity, py::arg("other"))
      .def("reset", &qmitigate::QuantumState::reset)
      .def("initialize_basis_state",
           &qmitigate::QuantumState::initialize_basis_state)
      .def("initialize_superposition",
           &qmitigate::QuantumState::initialize_superposition)
      .def("expectation_Z", &qmitigate::QuantumState::expectation_Z,
           py::arg("qubit"))
      .def("expectation_ZZ", &qmitigate::QuantumState::expectation_ZZ,
           py::arg("qubits"))
      .def("probability_zero", &qmitigate::QuantumState::probability_zero,
           py::arg("qubit"))
      .def("get_probabilities", &qmitigate::QuantumState::get_probabilities)
      .def("to_string", &qmitigate::QuantumState::to_string,
           py::arg("threshold") = 1e-6)
      .def("state_vector",
           [](const qmitigate::QuantumState &s) { return s.state_vector(); })
      .def("to_numpy",
           [](const qmitigate::QuantumState &s) {
             const auto &sv = s.state_vector();
             return py::array_t<std::complex<double>>(
                 {sv.size()}, {sizeof(std::complex<double>)}, sv.data());
           })
      .def("__repr__", [](const qmitigate::QuantumState &s) {
        return "<QuantumState qubits=" + std::to_string(s.num_qubits()) +
               " dim=" + std::to_string(s.dimension()) + ">";
      });

  // Factory functions
  m.def("create_bell_state", &qmitigate::create_bell_state);
  m.def("create_ghz_state", &qmitigate::create_ghz_state,
        py::arg("num_qubits"));
  m.def("create_w_state", &qmitigate::create_w_state, py::arg("num_qubits"));

  // =========================================================================
  // Gates Module
  // =========================================================================
  auto gates_module = m.def_submodule("gates", "Quantum gate operations");

  gates_module.def("apply_X", &qmitigate::gates::apply_X);
  gates_module.def("apply_Y", &qmitigate::gates::apply_Y);
  gates_module.def("apply_Z", &qmitigate::gates::apply_Z);
  gates_module.def("apply_H", &qmitigate::gates::apply_H);
  gates_module.def("apply_S", &qmitigate::gates::apply_S);
  gates_module.def("apply_T", &qmitigate::gates::apply_T);
  gates_module.def("apply_RX", &qmitigate::gates::apply_RX);
  gates_module.def("apply_RY", &qmitigate::gates::apply_RY);
  gates_module.def("apply_RZ", &qmitigate::gates::apply_RZ);
  gates_module.def("apply_CNOT", &qmitigate::gates::apply_CNOT);
  gates_module.def("apply_CZ", &qmitigate::gates::apply_CZ);
  gates_module.def("apply_SWAP", &qmitigate::gates::apply_SWAP);
  gates_module.def("apply_Toffoli", &qmitigate::gates::apply_Toffoli);

  // =========================================================================
  // Circuit
  // =========================================================================
  py::class_<qmitigate::Circuit>(m, "Circuit")
      .def(py::init<qmitigate::QubitIndex>(), py::arg("num_qubits"))
      .def_readonly("num_qubits", &qmitigate::Circuit::num_qubits)
      .def("h", &qmitigate::Circuit::h, py::arg("qubit"))
      .def("x", &qmitigate::Circuit::x, py::arg("qubit"))
      .def("y", &qmitigate::Circuit::y, py::arg("qubit"))
      .def("z", &qmitigate::Circuit::z, py::arg("qubit"))
      .def("rx", &qmitigate::Circuit::rx, py::arg("qubit"), py::arg("theta"))
      .def("ry", &qmitigate::Circuit::ry, py::arg("qubit"), py::arg("theta"))
      .def("rz", &qmitigate::Circuit::rz, py::arg("qubit"), py::arg("theta"))
      .def("cnot", &qmitigate::Circuit::cnot, py::arg("control"),
           py::arg("target"))
      .def("depth", &qmitigate::Circuit::depth)
      .def("__repr__", [](const qmitigate::Circuit &c) {
        return "<Circuit qubits=" + std::to_string(c.num_qubits) +
               " depth=" + std::to_string(c.depth()) + ">";
      });

  // =========================================================================
  // NoiseModel
  // =========================================================================
  py::class_<qmitigate::NoiseModel>(m, "NoiseModel")
      .def(py::init<>())
      .def("set_scale_factor", &qmitigate::NoiseModel::set_scale_factor)
      .def("scale_factor", &qmitigate::NoiseModel::scale_factor)
      .def("has_noise", &qmitigate::NoiseModel::has_noise)
      .def("seed", &qmitigate::NoiseModel::seed);

  m.def("create_depolarizing_noise_model",
        &qmitigate::create_depolarizing_noise_model,
        py::arg("error_probability"),
        "Create a noise model with depolarizing errors");

  // =========================================================================
  // Simulator
  // =========================================================================
  py::class_<qmitigate::Simulator>(m, "Simulator")
      .def(py::init<>())
      .def(py::init<qmitigate::NoiseModel>(), py::arg("noise_model"))
      .def("run", py::overload_cast<const qmitigate::Circuit &>(
                      &qmitigate::Simulator::run, py::const_))
      .def("expectation_Z", &qmitigate::Simulator::expectation_Z,
           py::arg("circuit"), py::arg("qubit"), py::arg("shots") = 1000)
      .def("set_noise_scale", &qmitigate::Simulator::set_noise_scale)
      .def("noise_scale", &qmitigate::Simulator::noise_scale)
      .def("seed", &qmitigate::Simulator::seed);

  // ZNE Functions
  m.def("fold_circuit_global", &qmitigate::fold_circuit_global,
        py::arg("circuit"), py::arg("scale_factor"),
        "Fold a circuit to scale effective noise (for ZNE)");
}
