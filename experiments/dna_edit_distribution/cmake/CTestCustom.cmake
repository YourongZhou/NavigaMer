# WFA2 registers a test for its optional align_benchmark executable
# unconditionally. This experiment links only the static library and excludes
# that executable from the default build, so the upstream CLI test is outside
# this project's test set.
list(APPEND CTEST_CUSTOM_TESTS_IGNORE wfa2lib)
