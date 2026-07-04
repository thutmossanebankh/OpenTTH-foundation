

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. UTILITAIRES DE ROTATION 3D SO(3)
# ==========================================
def rotation_matrix_3d(axis, theta):
    """Génère une matrice de rotation 3D autour d'un axe principal ('x', 'y' ou 'z')."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, cos_t, -sin_t], [0, sin_t, cos_t]])
    elif axis == 'y':
        return np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
    elif axis == 'z':
        return np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])

# ==========================================
# 2. CLASSE TTH TOPOLOGIE 3D
# ==========================================
class TTH_Topology_3D:
    def __init__(self, centers, rotations, lambdas):
        self.centers = np.array(centers)
        self.rotations = rotations
        self.lambdas = lambdas
        self.k = len(centers)

    def evaluate(self, point):
        p = np.array(point)
        result = np.zeros(3)
        for i in range(self.k):
            # Calcul du poids local lambda_i(x)
            w_i = self.lambdas[i](p)
            
            # F_i(x) = R_i(x - O_i) + O_i
            diff = p - self.centers[i]
            rotated_diff = np.dot(self.rotations[i], diff)
            F_i = rotated_diff + self.centers[i]
            
            # Combinaison convexe spatiale
            result += w_i * F_i
        return result

# ==========================================
# 3. CONFIGURATION DES CHAMPS DE DÉFORMATION
# ==========================================
# Foyer 1 : Rotation autour de l'axe Z
O1 = [-1.5, 0, 0]
R1 = rotation_matrix_3d('z', np.pi/2)

# Foyer 2 : Rotation autour de l'axe X (Crée une torsion croisée complexe)
O2 = [1.5, 0, 0]
R2 = rotation_matrix_3d('x', np.pi/2)

centers = [O1, O2]
rotations = [R1, R2]

# Champs gaussiens 3D
def l1(p): return np.exp(-0.3 * np.sum((p - O1)**2))
def l2(p): return np.exp(-0.3 * np.sum((p - O2)**2))

# Partition de l'unité
def norm_l1(p): return l1(p) / (l1(p) + l2(p) + 1e-9)
def norm_l2(p): return l2(p) / (l1(p) + l2(p) + 1e-9)

tth_3d = TTH_Topology_3D(centers, rotations, [norm_l1, norm_l2])

# ==========================================
# 4. GÉNÉRATION DU SOLIDE (CUBE) ET APPLICATION
# ==========================================
def generate_cube_edges(size, resolution):
    """Génère les points le long des 12 arêtes d'un cube centré sur l'origine."""
    points = []
    r = np.linspace(-size/2, size/2, resolution)
    s = size/2
    # Lignes parallèles à X
    for y in [-s, s]:
        for z in [-s, s]:
            points.extend([[x, y, z] for x in r])
    # Lignes parallèles à Y
    for x in [-s, s]:
        for z in [-s, s]:
            points.extend([[x, y, z] for y in r])
    # Lignes parallèles à Z
    for x in [-s, s]:
        for y in [-s, s]:
            points.extend([[x, y, z] for z in r])
    return np.array(points)

print("Génération de la topologie 3D et déformation du cube...")
resolution_arêtes = 30 # Nombre de points par arête pour voir la courbure
cube_initial = generate_cube_edges(size=2.0, resolution=resolution_arêtes)

cube_transforme = np.zeros_like(cube_initial)
for i, pt in enumerate(cube_initial):
    cube_transforme[i] = tth_3d.evaluate(pt)

# ==========================================
# 5. VISUALISATION 3D
# ==========================================
fig = plt.figure(figsize=(14, 7))
plt.suptitle("TTH-Topology : Action Tridimensionnelle sur un Solide", fontweight='bold', fontsize=15)

# Vue 1 : Cube Euclidien Initial
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Espace $\\mathbb{R}^3$ Initial (Arêtes Rectilignes)")
ax1.scatter(cube_initial[:,0], cube_initial[:,1], cube_initial[:,2], c='black', s=2, alpha=0.5)
ax1.scatter([O1[0], O2[0]], [O1[1], O2[1]], [O1[2], O2[2]], c='red', marker='x', s=100, label='Foyers $O_i$')
ax1.set_xlim([-3, 3]); ax1.set_ylim([-3, 3]); ax1.set_zlim([-3, 3])
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()

# Vue 2 : Variété Déformée (Cube Tordu)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Espace Fibré $\\mathcal{M}$ (Variété Courbe)")
# Coloration en fonction de la coordonnée Z initiale pour voir le brassage
ax2.scatter(cube_transforme[:,0], cube_transforme[:,1], cube_transforme[:,2], c=cube_initial[:,2], cmap='coolwarm', s=5, alpha=0.8)
ax2.scatter([O1[0], O2[0]], [O1[1], O2[1]], [O1[2], O2[2]], c='red', marker='x', s=100)
ax2.set_xlim([-3, 3]); ax2.set_ylim([-3, 3]); ax2.set_zlim([-3, 3])
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()

print("\n--- Validation Topologique ---")
print("1. Les droites de l'espace euclidien (arêtes du cube) se sont courbées.")
print("2. L'intersection des champs de rotation orthogonaux (X et Z) génère une torsion tridimensionnelle complexe.")