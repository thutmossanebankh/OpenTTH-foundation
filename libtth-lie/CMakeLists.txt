
cmake_minimum_required(VERSION 3.10)
project(libtth-lie)

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

add_library(tth_lie MODULE src/tth_lie.cpp src/bindings.cpp)
set_target_properties(tth_lie PROPERTIES PREFIX "" SUFFIX ".so")
target_compile_options(tth_lie PRIVATE -O3 -march=native -fPIC)