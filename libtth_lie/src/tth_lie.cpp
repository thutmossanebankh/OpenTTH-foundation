

#include "../include/tth_lie.hpp"
#include <stdexcept>

namespace tth {

    TTH_Operator::TTH_Operator(const std::vector<Vector2D>& O,
                               const std::vector<double>& angles,
                               const std::vector<ScalarField>& l)
        : centers(O), lambdas(l) {
        
        if (centers.size() != angles.size() || centers.size() != lambdas.size()) {
            throw std::invalid_argument("Incoherence dimensionnelle de l'operateur TTH.");
        }

        for (double theta : angles) {
            rotations.push_back({std::cos(theta), -std::sin(theta),
                                 std::sin(theta),  std::cos(theta)});
        }
    }

    Vector2D TTH_Operator::evaluate(const Vector2D& x) const {
        Vector2D result = {0.0, 0.0};
        for (size_t i = 0; i < centers.size(); ++i) {
            // F_i(x) = R_i(x - O_i) + O_i
            double dx = x[0] - centers[i][0];
            double dy = x[1] - centers[i][1];
            
            double rx = rotations[i][0]*dx + rotations[i][1]*dy;
            double ry = rotations[i][2]*dx + rotations[i][3]*dy;
            
            rx += centers[i][0];
            ry += centers[i][1];

            double w = lambdas[i](x);
            result[0] += w * rx;
            result[1] += w * ry;
        }
        return result;
    }

    Vector2D LieAlgebra::commutator(const TTH_Operator& T1, const TTH_Operator& T2, const Vector2D& x) {
        // T1_circ_T2 = T1(T2(x))
        Vector2D T2_x = T2.evaluate(x);
        Vector2D T1_circ_T2 = T1.evaluate(T2_x);
        
        // T2_circ_T1 = T2(T1(x))
        Vector2D T1_x = T1.evaluate(x);
        Vector2D T2_circ_T1 = T2.evaluate(T1_x);
        
        // [T1, T2](x) = T1(T2(x)) - T2(T1(x))
        return {T1_circ_T2[0] - T2_circ_T1[0], T1_circ_T2[1] - T2_circ_T1[1]};
    }

} // namespace tth