#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../include/tth_diff.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tth_diff, m) {
    m.doc() = "Moteur Differentiel et Riemannien de la Transformation de Thutmos";

    py::class_<tth::TTH_Differential>(m, "TTH_Differential")
        .def(py::init<const std::vector<tth::Vector2D>&,
                      const std::vector<double>&,
                      const std::vector<tth::TTH_Differential::ScalarField>&,
                      const std::vector<tth::TTH_Differential::GradientField>&>(),
             py::arg("centers"), py::arg("angles"), py::arg("lambdas"), py::arg("grad_lambdas"))
        
        .def("evaluate", &tth::TTH_Differential::evaluate, py::arg("x"))
        .def("compute_jacobian", &tth::TTH_Differential::compute_jacobian, py::arg("x"))
        .def("compute_metric_tensor", &tth::TTH_Differential::compute_metric_tensor, py::arg("x"));
}
