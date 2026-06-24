# libtth-core : Affine and Barycentric Engine (C++)

`libtth-core` is the fundamental engine of the Thutmos Transformation. It implements the elementary TTH defined as a convex combination of eccentric rotations.

## ⚙️ Features

* Ultra-fast evaluation of the affine operator $\mathcal{T}_\theta(P) = A(\theta)P + b(\theta)$.
* Exploitation of the isomorphism between the Euclidean plane $\mathbb{R}^2$ and the complex plane $\mathbb{C}$.
* Calculation of the effective eigenvalue $\lambda(\theta)$ to determine the contracting or isometric regime.
* Determination of the translation vector $b(\theta)$ encoding the perturbed barycentric influence.

## 🛠️ Compilation

```bash
cd build
cmake ..
make -j12

```

---

**Author:** Salvador-Jose Mountanta Famorosa

**License:** Apache 2.0
