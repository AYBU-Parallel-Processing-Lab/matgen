target_sources(
  matgen
  PRIVATE
    "bilinear_cuda.cu"
    "nearest_neighbor_cuda.cu"
    "lanczos_cuda.cu"
)
