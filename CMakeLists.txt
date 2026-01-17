cmake_minimum_required(VERSION 3.22)

# =============================================================================
# Project Options
# =============================================================================
option(MATGEN_ENABLE_OPENMP "Enable OpenMP support for shared-memory parallelization" ON)
option(MATGEN_ENABLE_MPI "Enable MPI support for distributed-memory parallelization" ON)
option(MATGEN_ENABLE_CUDA "Enable CUDA support for GPU acceleration" ON)
option(MATGEN_ENABLE_COMPILER_WARNINGS "Enable compiler warnings" ON)

# =============================================================================
# Project Definition
# ============================================================================
project(
  matgen
  VERSION 0.1.0
  DESCRIPTION "Parallel Sparse Matrix Generation with Nearest Neighbor Search"
  LANGUAGES C CXX
)

# =============================================================================
# C/C++ Standards
# =============================================================================
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
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
  set(OpenMP_RUNTIME_MSVC "llvm")
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_C_VERSION}")
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
  # Allow unsupported compilers (e.g., newer versions of GCC/Clang/MSVC)
  set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
  enable_language(CUDA)

  find_package(CUDAToolkit REQUIRED)

  set(MATGEN_HAS_CUDA TRUE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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
  target_compile_definitions(matgen PUBLIC MATGEN_HAS_CUDA)
  set_target_properties(matgen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(matgen PUBLIC CUDA::cudart)
endif()

# =============================================================================
# Link Standard Libraries
# ============================================================================

if (NOT MSVC)
  target_link_libraries(
    matgen
    PUBLIC
      m
  )
endif()

# =============================================================================
# Compiler Warnings
# =============================================================================
if (MATGEN_ENABLE_COMPILER_WARNINGS)
  if(MSVC)
    target_compile_options(matgen PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>)
  else()
    target_compile_options(matgen PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>)
  endif()
endif()

# =============================================================================
# Add Subdirectories
# =============================================================================
add_subdirectory("src")
add_subdirectory("testbed")

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
  message(STATUS "    Version:      ${CUDA_VERSION}")
endif()
