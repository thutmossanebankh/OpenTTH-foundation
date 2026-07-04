cmake_minimum_required(VERSION 3.10)
project(libtth-core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ajout du dossier include aux en-têtes
include_directories(include)

# Découverte de Python et de pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE pybind11_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(pybind11_INCLUDE_DIRS)
add_compile_options(${pybind11_INCLUDE_DIRS})

# Création du module Python
add_library(tth_core MODULE src/tth_core.cpp src/bindings.cpp)

# LA CORRECTION EST ICI : On force le nom exact "tth_core.so"
set_target_properties(tth_core PROPERTIES PREFIX "" SUFFIX ".so")

# Optimisation CPU
target_compile_options(tth_core PRIVATE -O3 -march=native -fPIC)