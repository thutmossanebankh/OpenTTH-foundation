
import tth_core
import numpy as np
import matplotlib.pyplot as plt

def plot_complex_polygon(ax, points, color, label, linestyle='-'):
    """Trace un polygone fermé à partir d'une liste de nombres complexes."""
    pts = points + [points[0]] # Fermer le polygone
    x = [p.real for p in pts]
    y = [p.imag for p in pts]
    ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=2)
    ax.scatter(x, y, color=color)

# 1. Configuration initiale (Triangle Rectangle)
# A = Origine, B = sur l'axe X, C = sur l'axe Y
A, B, C = 0j, 4+0j, 0+3j
triangle = [A, B, C]

# 2. Définition des Centres d'action (Foyers)
# Pour illustrer le multi-centrisme, on place 3 centres extérieurs
O1 = -1 - 1j
O2 =  5 - 1j
O3 =  2 + 4j
centers = [O1, O2, O3]

# 3. Paramètres de la TTH Élémentaire
weights = [1/3, 1/3, 1/3] # Combinaison convexe uniforme
phases = [0.0, 0.0, 0.0]  # Mouvement synchrone

# Initialisation du moteur C++
tth_engine = tth_core.TTH_Elementary(centers, weights, phases)

# 4. Génération de la figure
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
ax.set_aspect('equal')
plt.title("Transformation de Thutmos (TTH) - PoC Affine C++", fontsize=14, fontweight='bold')

# Trace les foyers
for i, O in enumerate(centers):
    ax.scatter([O.real], [O.imag], color='red', marker='x', s=100)
    ax.annotate(f" O{i+1}", (O.real, O.imag), fontsize=12)

# Trace le triangle initial
plot_complex_polygon(ax, triangle, 'black', 'Figure Initiale (θ=0)')

# 5. Balayage angulaire theta (Mouvement continu)
thetas = np.linspace(0, 2 * np.pi, 6) # On génère 6 images discrètes de la rotation
colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))

for i, theta in enumerate(thetas[1:]): # On ignore 0 (déjà tracé)
    transformed_triangle = [tth_engine.evaluate(P, theta) for P in triangle]
    plot_complex_polygon(ax, transformed_triangle, colors[i], f"θ = {theta:.2f}", linestyle='--')

# 6. Extraction des propriétés de l'opérateur (Validation analytique)
lambda_val = tth_engine.get_eigenvalue(np.pi/4)
b_val = tth_engine.get_translation(np.pi/4)
print(f"--- Propriétés de l'Opérateur pour θ=π/4 ---")
print(f"Valeur propre effective λ : {lambda_val:.3f}")
print(f"Module (Homothétie) |λ|   : {abs(lambda_val):.3f}")
print(f"Vecteur de translation b  : {b_val:.3f}")

plt.legend()
plt.grid(True, linestyle=':')
plt.xlabel("Axe Réel")
plt.ylabel("Axe Imaginaire")
plt.show()