

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../include/tth_lie.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tth_lie, m) {
    m.doc() = "Moteur d'Algebre de Lie pour la Transformation de Thutmos";

    py::class_<tth::TTH_Operator>(m, "TTH_Operator")
        .def(py::init<const std::vector<tth::Vector2D>&,
                      const std::vector<double>&,
                      const std::vector<tth::ScalarField>&>())
        .def("evaluate", &tth::TTH_Operator::evaluate);

    m.def("commutator", &tth::LieAlgebra::commutator, 
          "Calcule le commutateur [T1, T2](x)",
          py::arg("T1"), py::arg("T2"), py::arg("x"));
}