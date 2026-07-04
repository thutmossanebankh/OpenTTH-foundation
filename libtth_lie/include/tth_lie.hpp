
#ifndef TTH_LIE_HPP
#define TTH_LIE_HPP

#include <vector>
#include <array>
#include <functional>
#include <cmath>

namespace tth {

    using Vector2D = std::array<double, 2>;
    using ScalarField = std::function<double(const Vector2D&)>;

    // Représente une transformation TTH complète (un opérateur)
    class TTH_Operator {
    private:
        std::vector<Vector2D> centers;
        std::vector<std::array<double, 4>> rotations; // Matrices 2x2
        std::vector<ScalarField> lambdas;

    public:
        TTH_Operator(const std::vector<Vector2D>& O,
                     const std::vector<double>& angles,
                     const std::vector<ScalarField>& l);

        // Évaluation de T(x)
        Vector2D evaluate(const Vector2D& x) const;
    };

    // Moteur de l'Algèbre de Lie
    class LieAlgebra {
    public:
        // Calcule le crochet de Lie: [T1, T2](x) = T1(T2(x)) - T2(T1(x))
        static Vector2D commutator(const TTH_Operator& T1, const TTH_Operator& T2, const Vector2D& x);
    };

} // namespace tth

#endif // TTH_LIE_HPP
