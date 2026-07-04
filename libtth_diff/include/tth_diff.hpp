
#ifndef TTH_DIFF_HPP
#define TTH_DIFF_HPP

#include <vector>
#include <array>
#include <functional>
#include <cmath>

namespace tth {

    // Définition de structures algébriques légères pour R2
    using Vector2D = std::array<double, 2>;
    using Matrix2x2 = std::array<double, 4>; // [m00, m01, m10, m11]

    class TTH_Differential {


    public:
        // Types pour les fonctions de champs scalaires et leurs gradients
        using ScalarField = std::function<double(const Vector2D&)>;
        using GradientField = std::function<Vector2D(const Vector2D&)>;


    private:
        std::vector<Vector2D> centers;       // O_i
        std::vector<Matrix2x2> rotations;    // R_i dans SO(2)
        std::vector<ScalarField> lambdas;    // lambda_i(x)
        std::vector<GradientField> grad_lambdas; // nabla lambda_i(x)

    public:
        TTH_Differential(const std::vector<Vector2D>& O,
                         const std::vector<double>& angles,
                         const std::vector<ScalarField>& l,
                         const std::vector<GradientField>& dl);

        // Transformation de l'espace euclidien T(x)
        Vector2D evaluate(const Vector2D& x) const;

        // Calcul du Jacobien Analytique J_T(x)
        Matrix2x2 compute_jacobian(const Vector2D& x) const;

        // Calcul du Tenseur Métrique Induit g_ij(x)
        Matrix2x2 compute_metric_tensor(const Vector2D& x) const;
    };

} // namespace tth

#endif // TTH_DIFF_HPP
