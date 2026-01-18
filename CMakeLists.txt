cmake_minimum_required(VERSION 3.16)

# =============================================================================
# QMitigate: High-Performance Quantum Error Mitigation Engine
# =============================================================================
project(QMitigate
    VERSION 1.0.0
    DESCRIPTION "High-performance C++ quantum state-vector simulator with Zero-Noise Extrapolation"
    LANGUAGES CXX
)

# =============================================================================
# C++ Standard & Compiler Configuration
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =============================================================================
# Compiler Warnings (Research-Grade Code Quality)
# =============================================================================
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wduplicated-cond -Wduplicated-branches -Wlogical-op)
endif()

# =============================================================================
# Optimization Flags
# =============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# =============================================================================
# OpenMP (Critical for HPC Performance)
# =============================================================================
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

# =============================================================================
# PyBind11 (Python Bindings)
# =============================================================================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CMake, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# =============================================================================
# Google Test (Unit Testing)
# =============================================================================
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# =============================================================================
# Core Library: QMitigate Engine
# =============================================================================
add_library(qmitigate_core STATIC
    src/quantum_state.cpp
    src/gates.cpp
    src/noise_model.cpp
    src/simulator.cpp
)

target_include_directories(qmitigate_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(qmitigate_core
    PUBLIC
        OpenMP::OpenMP_CXX
)

# =============================================================================
# Python Module: qmitigate_cpp
# =============================================================================
pybind11_add_module(qmitigate_cpp
    src/python_bindings.cpp
)

target_link_libraries(qmitigate_cpp
    PRIVATE
        qmitigate_core
)

# Set the output name for the Python module
set_target_properties(qmitigate_cpp PROPERTIES
    OUTPUT_NAME "qmitigate_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/qmitigate"
)

# =============================================================================
# Unit Tests
# =============================================================================
if(BUILD_TESTS)
    add_executable(test_quantum_state
        tests/unit/test_quantum_state.cpp
    )
    target_link_libraries(test_quantum_state
        PRIVATE
            qmitigate_core
            GTest::gtest_main
    )

    add_executable(test_gates
        tests/unit/test_gates.cpp
    )
    target_link_libraries(test_gates
        PRIVATE
            qmitigate_core
            GTest::gtest_main
    )

    add_executable(test_noise_model
        tests/unit/test_noise_model.cpp
    )
    target_link_libraries(test_noise_model
        PRIVATE
            qmitigate_core
            GTest::gtest_main
    )

    add_executable(test_simulator
        tests/unit/test_simulator.cpp
    )
    target_link_libraries(test_simulator
        PRIVATE
            qmitigate_core
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(test_quantum_state)
    gtest_discover_tests(test_gates)
    gtest_discover_tests(test_noise_model)
    gtest_discover_tests(test_simulator)
endif()

# =============================================================================
# Benchmarks
# =============================================================================
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_executable(benchmark_engine
        benchmarks/benchmark_engine.cpp
    )
    target_link_libraries(benchmark_engine
        PRIVATE
            qmitigate_core
    )
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS qmitigate_core
    EXPORT QMitigateTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

# =============================================================================
# Package Configuration
# =============================================================================
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/QMitigateConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/QMitigateConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/QMitigateConfig.cmake"
    INSTALL_DESTINATION lib/cmake/QMitigate
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== QMitigate Build Configuration ===")
message(STATUS "Version:        ${PROJECT_VERSION}")
message(STATUS "Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP:         ${OpenMP_CXX_VERSION}")
message(STATUS "Python:         ${Python3_VERSION}")
message(STATUS "Build Tests:    ${BUILD_TESTS}")
message(STATUS "Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "=====================================")
message(STATUS "")
