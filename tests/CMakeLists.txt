add_executable(matgen_test)

if(MSVC)
  target_compile_options(matgen_test PRIVATE /EHsc)
endif()

add_subdirectory("src")

target_link_libraries(
  matgen_test
  PRIVATE
    matgen
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matgen_test)
