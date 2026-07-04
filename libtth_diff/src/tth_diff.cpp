

#include "../include/tth_diff.hpp"
#include <stdexcept>

namespace tth {

    TTH_Differential::TTH_Differential(const std::vector<Vector2D>& O,
                                       const std::vector<double>& angles,
                                       const std::vector<ScalarField>& l,
                                       const std::vector<GradientField>& dl)
        : centers(O), lambdas(l), grad_lambdas(dl) {
        
        if (centers.size() != angles.size() || centers.size() != lambdas.size()) {
            throw std::invalid_argument("Incoherence dimensionnelle des parametres multi-centriques.");
        }

        // Pre-calcul des matrices de rotation R_i
        for (double theta : angles) {
            rotations.push_back({std::cos(theta), -std::sin(theta),
                                 std::sin(theta),  std::cos(theta)});
        }
    }

    // Helper: produit matrice x vecteur
    Vector2D mat_vec_mul(const Matrix2x2& M, const Vector2D& v) {
        return {M[0]*v[0] + M[1]*v[1], M[2]*v[0] + M[3]*v[1]};
    }

    Vector2D TTH_Differential::evaluate(const Vector2D& x) const {
        Vector2D result = {0.0, 0.0};
        
        for (size_t i = 0; i < centers.size(); ++i) {
            Vector2D diff = {x[0] - centers[i][0], x[1] - centers[i][1]};
            Vector2D F_i = mat_vec_mul(rotations[i], diff);
            F_i[0] += centers[i][0];
            F_i[1] += centers[i][1];

            double w = lambdas[i](x);
            result[0] += w * F_i[0];
            result[1] += w * F_i[1];
        }
        return result;
    }

    Matrix2x2 TTH_Differential::compute_jacobian(const Vector2D& x) const {
        Matrix2x2 J = {0.0, 0.0, 0.0, 0.0};

        for (size_t i = 0; i < centers.size(); ++i) {
            Vector2D diff = {x[0] - centers[i][0], x[1] - centers[i][1]};
            Vector2D F_i = mat_vec_mul(rotations[i], diff);
            F_i[0] += centers[i][0];
            F_i[1] += centers[i][1];

            Vector2D grad = grad_lambdas[i](x);
            double w = lambdas[i](x);

            // Terme non-lineaire: produit tensoriel (dyadique) nabla(lambda_i) (x) F_i
            J[0] += grad[0] * F_i[0] + w * rotations[i][0]; // dx / dx
            J[1] += grad[1] * F_i[0] + w * rotations[i][1]; // dx / dy
            J[2] += grad[0] * F_i[1] + w * rotations[i][2]; // dy / dx
            J[3] += grad[1] * F_i[1] + w * rotations[i][3]; // dy / dy
        }
        return J;
    }

    Matrix2x2 TTH_Differential::compute_metric_tensor(const Vector2D& x) const {
        Matrix2x2 J = compute_jacobian(x);
        
        // La metrique g_ij = J^T * J (Pull-back de la metrique euclidienne)
        Matrix2x2 g;
        g[0] = J[0]*J[0] + J[2]*J[2]; // g_11
        g[1] = J[0]*J[1] + J[2]*J[3]; // g_12
        g[2] = J[1]*J[0] + J[3]*J[2]; // g_21 (symetrique)
        g[3] = J[1]*J[1] + J[3]*J[3]; // g_22
        
        return g;
    }

} // namespace tth