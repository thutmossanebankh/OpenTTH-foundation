# libtth-diff : TTH Differential Geometry (C++)

`libtth-diff` shifts the TTH theory into the domain of pure differential geometry by introducing spatially dependent weight functions $\lambda_i(x)$.

## ⚙️ Features

* Implementation of the analytical Jacobian tensor $J_T(x) = \sum_{i=1}^k \left( \nabla \lambda_i(x) \otimes F_i(x) + \lambda_i(x) R_i \right)$.
* Calculation of the non-linear deformation (transformation of Euclidean straight lines into curves).
* Riemannian pull-back for the generation of the local metric $g_{ij}(x)$.

## 🛠️ Compilation

The Python/C++ bridge uses `pybind11`.

```bash
mkdir build && cd build
cmake ..
make -j12

```

---

**Author:** Salvador-Jose Mountanta Famorosa

**License:** Apache 2.0
