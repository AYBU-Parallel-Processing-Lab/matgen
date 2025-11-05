# MatGen - Parallel Sparse Matrix Generation with Nearest Neighbor Search

A high-performance C/C++ implementation of sparse matrix generation, manipulation, and nearest neighbor search algorithms with parallel computing support (MPI, OpenMP, CUDA).

## 📚 Research Foundation

This project implements algorithms and concepts from the following research papers:

1. **Agarwal, A., Dahleh, M., Shah, D., and Shen, D. (2021)**
   _Causal Matrix Completion_
   arXiv:2109.15154v1
   Focus: Matrix completion techniques for sparse matrices with causal structures

2. **Bruch, S., Nardini, F. M., Rulli, C., and Venturini, R. (2025)**
   _Efficient Sketching and Nearest Neighbor Search Algorithms for Sparse Vector Sets_
   arXiv:2509.24815v1
   Focus: Efficient nearest neighbor search on sparse vectors and matrices

## 🎯 Project Goals

This project extends the original MatGen sparse matrix generator with:

- **Parallel implementations** using MPI, OpenMP, and CUDA
- **Nearest neighbor search** algorithms for sparse matrices
- **Non-square matrix support** for rectangular sparse matrices
- **Realistic value prediction** in generated matrices
- **Advanced matrix operations** including denoising and feature extraction
- **Benchmarking framework** for performance evaluation

## 🚀 Implementation Roadmap

### Phase 1: Core Parallel Infrastructure

- [x] Project structure setup (CMake, C17/C++20)
- [x] Third-party dependency management
- [x] Parallel computing frameworks integration (OpenMP, MPI, CUDA)
- [x] Sparse matrix storage formats (CSR, COO)
- [ ] Parallel sparse matrix I/O (MPI)
- [ ] Memory-efficient matrix representation

### Phase 2: Parallel Matrix Generation

- [ ] **OpenMP Implementation**: Shared-memory parallel generation
  - [ ] Fourier-based generation
  - [ ] Wavelet-based generation
  - [ ] Graph-based generation

- [ ] **MPI Implementation**: Distributed-memory parallel generation
  - [ ] Domain decomposition strategies
  - [ ] Inter-process communication optimization

- [ ] **CUDA Implementation**: GPU-accelerated generation
  - [ ] Kernel optimization for matrix operations
  - [ ] Memory coalescing strategies

### Phase 3: Bilinear Interpolation & Scaling

- [ ] Sequential bilinear interpolation
- [ ] OpenMP parallel interpolation
- [ ] CUDA accelerated interpolation
- [ ] Non-square matrix scaling

### Phase 4: Nearest Neighbor Search (Paper 2)

- [ ] Efficient sketching algorithms for sparse vectors
- [ ] Approximate nearest neighbor search
- [ ] Distance computations for sparse matrices
- [ ] Query optimization techniques

### Phase 5: Matrix Completion (Paper 1)

- [ ] Causal matrix completion algorithms
- [ ] Missing value prediction
- [ ] Realistic value generation based on matrix structure

### Phase 6: Advanced Features

- [ ] **Parallel Structural Feature Extractor**
  - [ ] Sparsity patterns analysis
  - [ ] Connected components
  - [ ] Degree distributions
  - [ ] Spectral properties

- [ ] **Parallel Matrix Denoising**
  - [ ] Noise detection in sparse matrices
  - [ ] Filtering algorithms
  - [ ] Structure-preserving denoising

### Phase 7: Benchmarking & Validation

- [ ] Performance benchmarking suite
- [ ] Scalability tests (weak/strong scaling)
- [ ] Comparison with existing tools
- [ ] Validation against real-world matrices

## 🏗️ Project Structure

```
matgen/
├── include/                 # Public header files
│   ├── matgen/
│   │   ├── core/           # Core data structures
│   │   ├── generation/     # Matrix generation methods
│   │   ├── parallel/       # Parallel implementations
│   │   ├── nn_search/      # Nearest neighbor search
│   │   └── utils/          # Utility functions
├── src/                     # Implementation files
│   ├── core/               # Core functionality
│   ├── generation/         # Generation algorithms
│   ├── parallel/
│   │   ├── openmp/         # OpenMP implementations
│   │   ├── mpi/            # MPI implementations
│   │   └── cuda/           # CUDA implementations
│   ├── nn_search/          # Nearest neighbor algorithms
│   ├── features/           # Feature extraction
│   └── denoising/          # Denoising algorithms
├── tests/                   # Unit tests
├── benchmarks/             # Performance benchmarks
├── examples/               # Example programs
├── docs/                   # Documentation
│   ├── algorithms/         # Algorithm descriptions
│   ├── api/                # API documentation
│   └── papers/             # Paper implementations
├── old_source/             # Original Python implementation
└── CMakeLists.txt
```

## 🛠️ Build Requirements

### Essential Dependencies

- **C Compiler**: GCC 9.0+ or Clang 10.0+ (C17 support)
- **C++ Compiler**: GCC 9.0+ or Clang 10.0+ (C++20 support)
- **CMake**: 3.25 or higher
- **Make** or **Ninja**: Build system

### Parallel Computing Support

- **OpenMP**: 4.5 or higher (usually bundled with compiler)
- **MPI**: OpenMPI 4.0+ or MPICH 3.3+
- **CUDA**: 11.0+ (optional, for GPU acceleration)

### Optional Dependencies

- **BLAS/LAPACK**: For linear algebra operations
- **HDF5**: For matrix I/O
- **Google Test**: For unit testing
- **Google Benchmark**: For performance testing

## 🔧 Building the Project

### Using CMake Presets (Recommended)

This project uses CMake Presets for simplified building. All presets use the Ninja generator and output to `out/build/<preset-name>`.

#### Available Presets

**Windows (MSVC):**

- `windows-msvc-debug` - MSVC x64 Debug build
- `windows-msvc-release` - MSVC x64 Release build

**Windows (Clang-CL):**

- `windows-clang-debug` - Clang-CL x64 Debug build
- `windows-clang-release` - Clang-CL x64 Release build

#### Quick Start

```bash
# Configure with a preset
cmake --preset windows-msvc-release

# Build
cmake --build --preset windows-msvc-release

# Test (when tests are available)
ctest --preset windows-msvc-release
```

#### List Available Presets

```bash
# List all configure presets
cmake --list-presets

# List all build presets
cmake --build --list-presets

# List all test presets
ctest --list-presets

```

#### Manual Build (Alternative)

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Build Configuration Options

The following CMake options will be available for enabling parallel features:

#### OpenMP Support

```bash
# With presets
cmake --preset windows-msvc-release -DENABLE_OPENMP=ON
cmake --build --preset windows-msvc-release

# Manual
cmake -DENABLE_OPENMP=ON ..
```

#### MPI Support

```bash
# With presets
cmake --preset windows-msvc-release -DENABLE_MPI=ON
cmake --build --preset windows-msvc-release

# Manual
cmake -DENABLE_MPI=ON ..
```

#### CUDA Support

```bash
# With presets
cmake --preset windows-msvc-release -DENABLE_CUDA=ON
cmake --build --preset windows-msvc-release

# Manual
cmake -DENABLE_CUDA=ON ..
```

#### Build with All Parallel Features

```bash
# With presets
cmake --preset windows-msvc-release -DENABLE_OPENMP=ON -DENABLE_MPI=ON -DENABLE_CUDA=ON
cmake --build --preset windows-msvc-release

# Manual
cmake -DENABLE_OPENMP=ON -DENABLE_MPI=ON -DENABLE_CUDA=ON ..
```

### Build Output

Build artifacts are placed in:

- **With presets**: `out/build/<preset-name>/`
- **Manual build**: `build/`

Compile commands are automatically exported to `compile_commands.json` for IDE integration.
