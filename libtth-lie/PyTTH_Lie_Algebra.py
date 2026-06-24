

import numpy as np
import matplotlib.pyplot as plt
import tth_lie

# ==========================================
# 1. DÉFINITION DES OPÉRATEURS T1 ET T2
# ==========================================
# L'opérateur T1 est centré sur la gauche, avec une forte rotation horaire
centers_T1 = [[-2.0, 0.0], [0.0, 0.0]]
angles_T1 = [-np.pi/4, 0.0]

# L'opérateur T2 est centré sur la droite, avec une forte rotation anti-horaire
centers_T2 = [[0.0, 0.0], [2.0, 0.0]]
angles_T2 = [0.0, np.pi/4]

# Fonctions de poids locales (Gaussiennes)
def lambda1_T1(x): return np.exp(-((x[0] - centers_T1[0][0])**2 + (x[1] - centers_T1[0][1])**2))
def lambda2_T1(x): return np.exp(-((x[0] - centers_T1[1][0])**2 + (x[1] - centers_T1[1][1])**2))
def norm_l1_T1(x): return lambda1_T1(x) / (lambda1_T1(x) + lambda2_T1(x) + 1e-6)
def norm_l2_T1(x): return lambda2_T1(x) / (lambda1_T1(x) + lambda2_T1(x) + 1e-6)

def lambda1_T2(x): return np.exp(-((x[0] - centers_T2[0][0])**2 + (x[1] - centers_T2[0][1])**2))
def lambda2_T2(x): return np.exp(-((x[0] - centers_T2[1][0])**2 + (x[1] - centers_T2[1][1])**2))
def norm_l1_T2(x): return lambda1_T2(x) / (lambda1_T2(x) + lambda2_T2(x) + 1e-6)
def norm_l2_T2(x): return lambda2_T2(x) / (lambda1_T2(x) + lambda2_T2(x) + 1e-6)

# Initialisation des opérateurs C++
T1 = tth_lie.TTH_Operator(centers_T1, angles_T1, [norm_l1_T1, norm_l2_T1])
T2 = tth_lie.TTH_Operator(centers_T2, angles_T2, [norm_l1_T2, norm_l2_T2])

# ==========================================
# 2. PREUVE NUMÉRIQUE DE LA NON-COMMUTATIVITÉ
# ==========================================
# Prenons un point d'épreuve au hasard
pt_test = [1.0, 1.0]

# T1_circ_T2 = T1(T2(x))
pt_T2_first = T2.evaluate(pt_test)
pt_T1_circ_T2 = T1.evaluate(pt_T2_first)

# T2_circ_T1 = T2(T1(x))
pt_T1_first = T1.evaluate(pt_test)
pt_T2_circ_T1 = T2.evaluate(pt_T1_first)

comm_val = tth_lie.commutator(T1, T2, pt_test)

print("--- Théorème de Non-Commutativité ---")
print(f"Point Initial x        : {pt_test}")
print(f"Trajet T1(T2(x))       : [{pt_T1_circ_T2[0]:.4f}, {pt_T1_circ_T2[1]:.4f}]")
print(f"Trajet T2(T1(x))       : [{pt_T2_circ_T1[0]:.4f}, {pt_T2_circ_T1[1]:.4f}]")
print(f"Vecteur Commutateur    : [{comm_val[0]:.4f}, {comm_val[1]:.4f}]")

norm_comm = np.linalg.norm(comm_val)
if norm_comm > 1e-5:
    print(f"\n[SUCCÈS] T1 o T2 != T2 o T1. La transformation est non-commutative (Norme: {norm_comm:.4f}).")

# ==========================================
# 3. CHAMP DE VECTEURS DU CROCHET DE LIE
# ==========================================
print("\nCalcul du champ de vecteurs du commutateur [T1, T2] sur la grille...")
n_pts = 20
x_range = np.linspace(-3, 3, n_pts)
y_range = np.linspace(-3, 3, n_pts)
X, Y = np.meshgrid(x_range, y_range)

U = np.zeros_like(X) # Composante X du commutateur
V = np.zeros_like(X) # Composante Y du commutateur
Comm_Norm = np.zeros_like(X)

for i in range(n_pts):
    for j in range(n_pts):
        pt = [X[i,j], Y[i,j]]
        comm = tth_lie.commutator(T1, T2, pt)
        U[i,j] = comm[0]
        V[i,j] = comm[1]
        Comm_Norm[i,j] = np.linalg.norm(comm)

plt.figure(figsize=(10, 8))
plt.title("Algèbre de Lie TTH : Champ Vectoriel du Commutateur $[T_1, T_2]$", fontweight='bold', fontsize=14)

# Affiche la norme du commutateur en fond couleur (zones de forte non-commutativité)
contour = plt.contourf(X, Y, Comm_Norm, 30, cmap='magma', alpha=0.6)
plt.colorbar(contour, label="Norme du Crochet de Lie $|[T_1, T_2]|$")

# Trace le champ de vecteurs (Quiver plot)
plt.quiver(X, Y, U, V, color='white', scale=15, alpha=0.9)

# Trace les foyers des deux opérateurs
plt.scatter([c[0] for c in centers_T1], [c[1] for c in centers_T1], c='cyan', marker='o', s=100, edgecolors='black', label='Foyers T1')
plt.scatter([c[0] for c in centers_T2], [c[1] for c in centers_T2], c='lime', marker='s', s=100, edgecolors='black', label='Foyers T2')

plt.legend()
plt.xlabel("Axe X")
plt.ylabel("Axe Y")
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()