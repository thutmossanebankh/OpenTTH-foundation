
#include "../include/tth_core.hpp"
#include <numeric>

namespace tth {

    TTH_Elementary::TTH_Elementary(const std::vector<Point2D>& O, 
                                   const std::vector<double>& w, 
                                   const std::vector<double>& phi) 
        : centers(O), weights(w), phases(phi) {
        
        if (centers.size() != weights.size() || centers.size() != phases.size()) {
            throw std::invalid_argument("Les dimensions des centres, poids et phases doivent etre identiques.");
        }
        if (centers.size() < 3) {
            throw std::invalid_argument("La TTH necessite au moins n=3 centres.");
        }
        normalize_weights();
    }

    void TTH_Elementary::normalize_weights() {
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum <= 0.0) {
            throw std::invalid_argument("La somme des poids doit etre strictement positive.");
        }
        for (double& w : weights) {
            w /= sum;
        }
    }

    Point2D TTH_Elementary::evaluate(const Point2D& P, double theta) const {
        Point2D result(0.0, 0.0);
        for (size_t i = 0; i < centers.size(); ++i) {
            // Phasor complexe = exp(i * (theta + phi_i))
            Point2D rotation_phasor = std::polar(1.0, theta + phases[i]);
            
            // w_i * [ O_i + R(theta+phi_i)(P - O_i) ]
            result += weights[i] * (centers[i] + rotation_phasor * (P - centers[i]));
        }
        return result;
    }

    Point2D TTH_Elementary::get_eigenvalue(double theta) const {
        Point2D lambda(0.0, 0.0);
        for (size_t i = 0; i < weights.size(); ++i) {
            lambda += weights[i] * std::polar(1.0, theta + phases[i]);
        }
        return lambda; // Le module definit le facteur d'homothetie globale
    }

    Point2D TTH_Elementary::get_translation(double theta) const {
        Point2D b(0.0, 0.0);
        for (size_t i = 0; i < centers.size(); ++i) {
            Point2D rotation_phasor = std::polar(1.0, theta + phases[i]);
            b += weights[i] * (Point2D(1.0, 0.0) - rotation_phasor) * centers[i];
        }
        return b;
    }

    Point2D TTH_Elementary::get_fixed_point(double theta) const {
        Point2D lambda = get_eigenvalue(theta);
        
        // Verifie la condition de contractance |lambda| < 1
        if (std::abs(lambda) >= 1.0 - 1e-9) {
            throw std::domain_error("L'application n'est pas strictement contractante pour ce theta (|lambda| >= 1). Pas de point fixe unique.");
        }
        
        Point2D b = get_translation(theta);
        
        // P_infty = (1 - lambda)^(-1) * b
        return b / (Point2D(1.0, 0.0) - lambda);
    }

} // namespace tth
