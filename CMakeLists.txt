cmake_minimum_required(VERSION 3.25)

project(
  matgen
  VERSION 0.1.0
  DESCRIPTION "Parallel Sparse Matrix Generation with Nearest Neighbor Search"
  LANGUAGES C CXX CUDA
)

# =============================================================================
# Project Options
# =============================================================================
option(MATGEN_ENABLE_OPENMP "Enable OpenMP support for shared-memory parallelization" ON)
option(MATGEN_ENABLE_MPI "Enable MPI support for distributed-memory parallelization" ON)
option(MATGEN_ENABLE_CUDA "Enable CUDA support for GPU acceleration" ON)
option(MATGEN_BUILD_TESTS "Build unit tests" ON)
option(MATGEN_BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(MATGEN_BUILD_EXAMPLES "Build example programs" ON)

# =============================================================================
# C/C++ Standards
# =============================================================================
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Build Type Configuration
# =============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# =============================================================================
# Find Parallel Computing Libraries
# =============================================================================

# OpenMP
if(MATGEN_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_C_VERSION}")
    set(MATGEN_HAS_OPENMP TRUE)
  else()
    message(FATAL_ERROR "OpenMP requested but not found")
  endif()
endif()

# MPI
if(MATGEN_ENABLE_MPI)
  find_package(MPI REQUIRED)
  if(MPI_FOUND)
    message(STATUS "MPI found: ${MPI_C_VERSION}")
    set(MATGEN_HAS_MPI TRUE)
  else()
    message(FATAL_ERROR "MPI requested but not found")
  endif()
endif()

# CUDA
if(MATGEN_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    set(MATGEN_HAS_CUDA TRUE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  else()
    message(FATAL_ERROR "CUDA requested but not found")
  endif()
endif()

# =============================================================================
# Main Library Target
# =============================================================================
add_library(matgen STATIC)

target_include_directories(
  matgen
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_include_directories(
  matgen
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

# =============================================================================
# Link Parallel Libraries
# =============================================================================

# OpenMP
if(MATGEN_ENABLE_OPENMP AND OpenMP_FOUND)
  target_link_libraries(matgen PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  target_compile_definitions(matgen PUBLIC MATGEN_HAS_OPENMP)
endif()

# MPI
if(MATGEN_ENABLE_MPI AND MPI_FOUND)
  target_link_libraries(matgen PUBLIC MPI::MPI_C MPI::MPI_CXX)
  target_compile_definitions(matgen PUBLIC MATGEN_HAS_MPI)
  target_include_directories(matgen PUBLIC ${MPI_C_INCLUDE_DIRS})
endif()

# CUDA
if(MATGEN_ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_link_libraries(matgen PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
  target_compile_definitions(matgen PUBLIC MATGEN_HAS_CUDA)
  set_target_properties(matgen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# =============================================================================
# Compiler Warnings
# =============================================================================
if(MSVC)
  target_compile_options(matgen PRIVATE /W4 /WX-)
else()
  target_compile_options(matgen PRIVATE -Wall -Wextra -Wpedantic)
endif()

# =============================================================================
# Add Subdirectories
# =============================================================================
add_subdirectory("src")
add_subdirectory("third_party")

if(MATGEN_BUILD_EXAMPLES)
  add_subdirectory("examples")
endif()

if(MATGEN_BUILD_TESTS)
  enable_testing()
  add_subdirectory("tests")
endif()

if(MATGEN_BUILD_BENCHMARKS)
  add_subdirectory("benchmarks")
endif()

# =============================================================================
# Configuration Summary
# =============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "MatGen Configuration Summary")
message(STATUS "========================================")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler:       ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CXX Compiler:     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Parallel Features:")
message(STATUS "  OpenMP:         ${MATGEN_ENABLE_OPENMP}")
if(ENABLE_OPENMP)
  message(STATUS "    Version:      ${OpenMP_C_VERSION}")
endif()
message(STATUS "  MPI:            ${MATGEN_ENABLE_MPI}")
if(ENABLE_MPI)
  message(STATUS "    Version:      ${MPI_C_VERSION}")
endif()
message(STATUS "  CUDA:           ${MATGEN_ENABLE_CUDA}")
if(ENABLE_CUDA)
  message(STATUS "    Version:      ${CUDAToolkit_VERSION}")
  message(STATUS "    Arch:         ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
message(STATUS "Optional Features:")
message(STATUS "  Build Tests:    ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "========================================")
message(STATUS "")
