cmake_minimum_required(VERSION 3.10)
project(libtth-diff)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE pybind11_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(pybind11_INCLUDE_DIRS)
add_compile_options(${pybind11_INCLUDE_DIRS})

add_library(tth_diff MODULE src/tth_diff.cpp src/bindings.cpp)
set_target_properties(tth_diff PROPERTIES PREFIX "" SUFFIX ".so")
target_compile_options(tth_diff PRIVATE -O3 -march=native -fPIC)
