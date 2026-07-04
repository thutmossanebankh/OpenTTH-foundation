import tth_diff
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration Multi-centrique (2 foyers d'action)
centers = [[-2.0, 0.0], [2.0, 0.0]]
angles = [np.pi/4, -np.pi/4]  # Rotations opposées pour créer une torsion

# 2. Définition des champs scalaires spatiaux λ_i(x)
# Nous utilisons une transition douce (tangente hyperbolique) le long de l'axe X
# Cela garantit que λ1 + λ2 = 1 en tout point (Partition de l'unité)
def lambda1(x):
    return 0.5 - 0.5 * np.tanh(x[0])

def lambda2(x):
    return 0.5 + 0.5 * np.tanh(x[0])

# 3. Gradients analytiques ∇λ_i(x)
def grad_lambda1(x):
    dx = -0.5 * (1.0 - np.tanh(x[0])**2)
    return [dx, 0.0]

def grad_lambda2(x):
    dx = 0.5 * (1.0 - np.tanh(x[0])**2)
    return [dx, 0.0]

lambdas = [lambda1, lambda2]
grad_lambdas = [grad_lambda1, grad_lambda2]

# Initialisation du moteur différentiel C++
diff_engine = tth_diff.TTH_Differential(centers, angles, lambdas, grad_lambdas)

# --- ANALYSE ANALYTIQUE AU CENTRE (0,0) ---
print("=== Analyse de la Métrique Induite (Origine) ===")
x_test = [0.0, 0.0]
J = diff_engine.compute_jacobian(x_test)
g = diff_engine.compute_metric_tensor(x_test)

print(f"Jacobien J_T(0,0):")
print(f"[{J[0]:.4f}, {J[1]:.4f}]")
print(f"[{J[2]:.4f}, {J[3]:.4f}]\n")

print(f"Tenseur Métrique g_ij(0,0):")
print(f"g_11 = {g[0]:.4f} | g_12 = {g[1]:.4f}")
print(f"g_21 = {g[2]:.4f} | g_22 = {g[3]:.4f}")
# Si g_ij n'est pas la matrice identité [1, 0; 0, 1], l'espace est courbé !
if abs(g[0]-1.0) > 1e-5 or abs(g[3]-1.0) > 1e-5:
    print("\n[SUCCÈS] Métrique non-euclidienne détectée ! L'espace est courbe.")

# --- VISUALISATION DE LA DÉFORMATION DE L'ESPACE ---
# On crée une grille cartésienne euclidienne
grid_x, grid_y = np.meshgrid(np.linspace(-4, 4, 15), np.linspace(-4, 4, 15))
trans_x, trans_y = np.zeros_like(grid_x), np.zeros_like(grid_y)

# On applique la TTH à chaque point de la grille
for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        pt = [grid_x[i,j], grid_y[i,j]]
        transformed = diff_engine.evaluate(pt)
        trans_x[i,j] = transformed[0]
        trans_y[i,j] = transformed[1]

plt.figure(figsize=(12, 6))

# Grille Initiale (Euclidienne)
ax1 = plt.subplot(121)
ax1.set_title("Espace Euclidien Initial", fontweight='bold')
for i in range(grid_x.shape[0]):
    ax1.plot(grid_x[i,:], grid_y[i,:], 'k-', alpha=0.3)
    ax1.plot(grid_x[:,i], grid_y[:,i], 'k-', alpha=0.3)
ax1.scatter([-2, 2], [0, 0], c='red', marker='x', s=100, label='Foyers')
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':')

# Grille Transformée (Variété Riemannienne Induite)
ax2 = plt.subplot(122)
ax2.set_title("Variété TTH Induite (Courbure dynamique)", fontweight='bold')
for i in range(grid_x.shape[0]):
    ax2.plot(trans_x[i,:], trans_y[i,:], 'b-', alpha=0.6)
    ax2.plot(trans_x[:,i], trans_y[:,i], 'b-', alpha=0.6)
ax2.scatter([-2, 2], [0, 0], c='red', marker='x', s=100)
ax2.set_aspect('equal')
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.show()
