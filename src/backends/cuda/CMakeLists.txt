add_subdirectory("core")

set_property(
  TARGET matgen
  PROPERTY
    CUDA_SEPARABLE_COMPILATION ON
)
