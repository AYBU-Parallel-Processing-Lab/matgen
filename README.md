# MatGen - Parallel Sparse Matrix Scaling and Value Estimation

A high-performance C library for generating sparse matrices through parallel scaling algorithms (Nearest Neighbor and Bilinear Interpolation) with realistic value estimation. Implements OpenMP, MPI, and CUDA backends for scalability.

```
matgen/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
│
├── docs/                           # All documentation
│   ├── api/                        # API reference (future: Doxygen output)
│   ├── guides/                     # User guides and tutorials
│   │   ├── getting_started.md
│   │   ├── execution_policies.md
│   │   └── backends.md
│   └── design/                     # Design documents
│       ├── architecture.md
│       └── execution_policy_design.md
│
├── include/matgen/                 # Public headers (what users include)
│   ├── matgen.h                    # Main convenience header
│   │
│   ├── core/                       # Core types and data structures
│   │   ├── types.h
│   │   ├── error.h
│   │   ├── matrix/                 # Matrix formats
│   │   │   ├── csr.h
│   │   │   ├── coo.h
│   │   │   └── conversion.h
│   │   └── execution/              # Execution policy system
│   │       ├── policy.h
│   │       └── dispatch.h
│   │
│   ├── algorithms/                 # Algorithm public interfaces
│   │   ├── scaling.h               # Unified scaling interface
│   │   └── estimation.h            # Value estimation (future)
│   │
│   ├── generators/                 # Matrix generators
│   │   └── random.h
│   │
│   ├── io/                         # Input/output
│   │   ├── mtx.h                   # Matrix Market format
│   │   └── formats.h               # Other formats (future)
│   │
│   ├── math/                       # Math utilities
│   │   └── constants.h
│   │
│   └── utils/                      # Utilities
│       ├── log.h
│       └── argparse.h
│
├── src/                            # Implementation (internal)
│   ├── CMakeLists.txt
│   │
│   ├── core/                       # Core implementations
│   │   ├── CMakeLists.txt
│   │   ├── matrix/
│   │   │   ├── csr.c
│   │   │   ├── coo.c
│   │   │   └── conversion.c
│   │   └── execution/
│   │       ├── policy.c
│   │       └── dispatch.c
│   │
│   ├── algorithms/                 # Algorithm implementations
│   │   ├── CMakeLists.txt
│   │   ├── scaling/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── scaling_common.h    # Shared utilities
│   │   │   ├── scaling_common.c
│   │   │   ├── scaling_dispatch.c  # Main dispatch logic
│   │   │   │
│   │   │   ├── backends/           # Backend-specific implementations
│   │   │   │   ├── seq/            # Sequential backend
│   │   │   │   │   ├── bilinear_seq.c
│   │   │   │   │   └── nearest_neighbor_seq.c
│   │   │   │   │
│   │   │   │   ├── openmp/         # OpenMP backend
│   │   │   │   │   ├── bilinear_omp.c
│   │   │   │   │   └── nearest_neighbor_omp.c
│   │   │   │   │
│   │   │   │   ├── cuda/           # CUDA backend
│   │   │   │   │   ├── bilinear_cuda.cu
│   │   │   │   │   ├── nearest_neighbor_cuda.cu
│   │   │   │   │   └── cuda_utils.cuh
│   │   │   │   │
│   │   │   │   └── mpi/            # MPI backend (future)
│   │   │   │       ├── bilinear_mpi.c
│   │   │   │       └── nearest_neighbor_mpi.c
│   │   │   │
│   │   │   └── internal/           # Internal headers (not public)
│   │   │       ├── bilinear_internal.h
│   │   │       └── nearest_neighbor_internal.h
│   │   │
│   │   └── estimation/             # Value estimation (future)
│   │       └── ...
│   │
│   ├── generators/
│   │   ├── CMakeLists.txt
│   │   └── random.c
│   │
│   ├── io/
│   │   ├── CMakeLists.txt
│   │   ├── mtx_reader.c
│   │   ├── mtx_writer.c
│   │   └── mtx_common.h
│   │
│   ├── math/
│   │   └── CMakeLists.txt
│   │
│   └── utils/
│       ├── CMakeLists.txt
│       ├── log.c
│       ├── argparse.c
│       └── triplet_buffer.c
│
├── examples/                       # Example programs (simple, focused)
│   ├── CMakeLists.txt
│   ├── 01_basic_scaling.c          # Simple scaling example
│   ├── 02_execution_policies.c     # Using execution policies
│   ├── 03_pipeline.c               # Pipeline example
│   ├── 04_custom_parameters.c      # Custom backend parameters
│   └── 05_io_operations.c          # Reading/writing matrices
│
├── apps/                           # Standalone applications
│   ├── CMakeLists.txt
│   ├── scale_cli/                  # CLI scaling tool
│   │   ├── main.c
│   │   └── README.md
│   └── benchmark_tool/             # Benchmarking tool (future)
│       └── main.c
│
├── tests/                          # Unit tests
│   ├── CMakeLists.txt
│   └── src/
│       ├── core/
│       │   ├── test_csr_matrix.c
│       │   ├── test_coo_matrix.c
│       │   ├── test_conversion.c
│       │   └── test_execution_policy.c
│       ├── algorithms/
│       │   ├── test_bilinear_seq.c
│       │   ├── test_bilinear_omp.c
│       │   ├── test_bilinear_cuda.c
│       │   └── test_scaling_dispatch.c
│       ├── generators/
│       │   └── test_random.c
│       ├── io/
│       │   └── test_mtx.c
│       └── utils/
│           └── test_log.c
│
├── benchmarks/                     # Performance benchmarks
│   ├── CMakeLists.txt
│   └── src/
│       ├── bench_scaling_backends.c
│       ├── bench_matrix_formats.c
│       └── bench_io.c
│
├── third_party/                    # External dependencies
│   └── ...
│
└── scripts/                        # Build/utility scripts
    ├── build.sh
    ├── test.sh
    └── benchmark.sh
```
