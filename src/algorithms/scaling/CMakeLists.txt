target_sources(
  matgen
  PRIVATE
    "bilinear.c"
    "nearest_neighbor.c"
)

if(MATGEN_ENABLE_OPENMP)
  target_sources(
    matgen
    PRIVATE
      "bilinear_omp.c"
      "nearest_neighbor_omp.c"
  )
endif()

if(MATGEN_ENABLE_CUDA)
  target_sources(
    matgen
    PRIVATE
      # "bilinear_cuda.cu"
      # "nearest_neighbor_cuda.cu"
  )
endif()
