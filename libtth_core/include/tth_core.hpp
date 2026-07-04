#ifndef TTH_CORE_HPP
#define TTH_CORE_HPP

#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>

namespace tth {

    // Utilisation de std::complex pour l'isomorphisme R2 <-> C
    using Point2D = std::complex<double>;

    class TTH_Elementary {
    private:
        std::vector<Point2D> centers; // Foyers d'action O_i
        std::vector<double> weights;  // Poids w_i
        std::vector<double> phases;   // Phases initiales phi_i

        // Vérifie que la somme des poids vaut 1 (combinaison convexe)
        void normalize_weights();

    public:
        // Constructeur
        TTH_Elementary(const std::vector<Point2D>& O, 
                       const std::vector<double>& w, 
                       const std::vector<double>& phi);

        // Applique la TTH sur un point P pour un angle de balayage theta
        Point2D evaluate(const Point2D& P, double theta) const;

        // Calcule la valeur propre complexe effective lambda(theta)
        Point2D get_eigenvalue(double theta) const;

        // Calcule le vecteur de translation b(theta)
        Point2D get_translation(double theta) const;

        // Calcule l'unique point fixe asymptotique P_infty
        Point2D get_fixed_point(double theta) const;
    };

} // namespace tth

#endif // TTH_CORE_HPP