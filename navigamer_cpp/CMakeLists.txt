cmake_minimum_required(VERSION 3.14)
project(NavigaMer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# NavigaMer v7 (Multilateration-Enhanced) - C++ 实现
add_executable(navigamer
  src/main.cpp
  src/structure.cpp
  src/tools.cpp
  src/index_builder.cpp
  src/search_engine.cpp
  src/io_utils.cpp
)

target_include_directories(navigamer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Release 优化
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(navigamer PRIVATE -O3 -DNDEBUG)
endif()
