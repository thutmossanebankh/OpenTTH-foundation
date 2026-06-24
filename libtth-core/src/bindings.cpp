
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "../include/tth_core.hpp"

namespace py = pybind11;

// Le nom du module défini ici ('tth_core') sera celui importé en Python
PYBIND11_MODULE(tth_core, m) {
    m.doc() = "Moteur noyau C++ de la Transformation de Thutmos (TTH)";

    py::class_<tth::TTH_Elementary>(m, "TTH_Elementary")
        // Constructeur
        .def(py::init<const std::vector<std::complex<double>>&, 
                      const std::vector<double>&, 
                      const std::vector<double>&>(),
             py::arg("centers"), py::arg("weights"), py::arg("phases"))
        
        // Méthodes
        .def("evaluate", &tth::TTH_Elementary::evaluate, 
             "Évalue la TTH pour un point P et un angle theta",
             py::arg("P"), py::arg("theta"))
        
        .def("get_eigenvalue", &tth::TTH_Elementary::get_eigenvalue,
             "Retourne la valeur propre complexe effective",
             py::arg("theta"))
             
        .def("get_translation", &tth::TTH_Elementary::get_translation,
             "Retourne le vecteur de translation",
             py::arg("theta"))
             
        .def("get_fixed_point", &tth::TTH_Elementary::get_fixed_point,
             "Calcule le point fixe asymptotique (si |lambda| < 1)",
             py::arg("theta"));
}