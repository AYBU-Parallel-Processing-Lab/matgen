add_executable(scale_cli)

target_sources(
  scale_cli
  PRIVATE
    "scale_cli.c"
)

target_link_libraries(
  scale_cli
  PRIVATE
    matgen
)
