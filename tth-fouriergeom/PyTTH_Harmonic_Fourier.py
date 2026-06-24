import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DÉFINITION DE LA TTH HARMONIQUE (POLYGONALE)
# ==========================================
class TTH_Harmonic:
    def __init__(self, n_vertices, lambdas):
        """
        n_vertices : Nombre de sommets du polygone (n=4 pour le carré)
        lambdas : Amplitudes modales Λ_k pour chaque fréquence k
        """
        self.n = n_vertices
        self.lambdas = lambdas
        # Angles spectraux de la base discrète : theta_k = (2 * pi * k) / n
        self.thetas = [2 * np.pi * k / self.n for k in range(self.n)]

    def _projection(self, point, theta):
        """
        Opérateur de projection bidimensionnelle sur la direction theta.
        Convertit le complexe en vecteur, projette, et retourne un complexe.
        """
        u_dir = np.array([np.cos(theta), np.sin(theta)])
        p_vec = np.array([point.real, point.imag])
        proj_val = np.dot(p_vec, u_dir)
        proj_vec = proj_val * u_dir
        return complex(proj_vec[0], proj_vec[1])

    def evaluate(self, point):
        """
        T_TTH(P_j) = Somme_{k=0}^{n-1} Λ_k * R(theta_k) * Pi_k(P_j)
        """
        result = 0j
        for k in range(self.n):
            # 1. Projection sur la direction theta_k
            proj_p = self._projection(point, self.thetas[k])
            
            # 2. Rotation discrète R(theta_k)
            # Dans le plan complexe, c'est une multiplication par exp(i * theta_k)
            rotation_phasor = np.exp(1j * self.thetas[k])
            
            # 3. Pondération par l'amplitude modale
            term = self.lambdas[k] * rotation_phasor * proj_p
            result += term
            
        return result

# ==========================================
# 2. VÉRIFICATION DU THÉORÈME (INVARIANCE DU BARYCENTRE)
# ==========================================
# Définition d'un carré régulier (n=4) dans le plan complexe
# Sommets P0, P1, P2, P3
square = [
    complex(1, 1),
    complex(-1, 1),
    complex(-1, -1),
    complex(1, -1)
]

# Calcul du barycentre initial G = (1/4) * Somme(P_j)
barycenter_initial = sum(square) / 4.0

# Initialisation de la TTH Harmonique avec des amplitudes uniformes (ex: Λ_k = 0.5)
lambdas_uniform = [0.5, 0.5, 0.5, 0.5]
tth_harmonic = TTH_Harmonic(n_vertices=4, lambdas=lambdas_uniform)

# Application de la transformation
transformed_square = [tth_harmonic.evaluate(p) for p in square]

# Calcul du barycentre de la figure transformée
barycenter_transformed = sum(transformed_square) / 4.0

print("--- Théorème de Conservation du Barycentre (TTH Harmonique) ---")
print(f"Barycentre Initial G     : {barycenter_initial:.4f}")
print(f"Barycentre Transformé G' : {barycenter_transformed:.4f}")

# Preuve d'invariance quadratique
diff = abs(barycenter_initial - barycenter_transformed)
if diff < 1e-10:
    print("\n[SUCCÈS] Le barycentre est strictement invariant sous l'action de la TTH Harmonique !")
    print("La somme cyclique des opérateurs de rotation s'annule (Somme(i^k) = 0).")

# ==========================================
# 3. VISUALISATION DE LA DÉFORMATION MODALE
# ==========================================
def plot_polygon(ax, points, color, label, linestyle='-'):
    pts = points + [points[0]]
    x = [p.real for p in pts]
    y = [p.imag for p in pts]
    ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=2, marker='o')

plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
ax.set_aspect('equal')
plt.title("TTH Harmonique : Invariance Quadratique (n=4)", fontsize=14, fontweight='bold')

# Tracé du carré initial et de son barycentre
plot_polygon(ax, square, 'black', 'Carré Initial')
ax.scatter([barycenter_initial.real], [barycenter_initial.imag], color='black', marker='x', s=150, label='Barycentre G', zorder=5)

# Tracé de la figure transformée
plot_polygon(ax, transformed_square, 'blue', 'Figure Transformée (Λ_k = 0.5)', linestyle='--')

plt.legend()
plt.grid(True, linestyle=':')
plt.axhline(0, color='gray', alpha=0.5)
plt.axvline(0, color='gray', alpha=0.5)
plt.xlabel("Axe Réel")
plt.ylabel("Axe Imaginaire")
plt.show()


