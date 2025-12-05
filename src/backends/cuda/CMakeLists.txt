add_subdirectory("core")
add_subdirectory("algorithms")

set_property(
  TARGET matgen
  PROPERTY
    CUDA_SEPARABLE_COMPILATION ON
)
