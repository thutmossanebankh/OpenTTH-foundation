

import tth_core
import numpy as np
import matplotlib.pyplot as plt

def generate_orbit(engine, P0, theta, iterations):
    """Génère l'orbite d'un point sous itérations successives de la TTH"""
    orbit = [P0]
    P = P0
    for _ in range(iterations):
        P = engine.evaluate(P, theta)
        orbit.append(P)
    return orbit

# ==========================================
# CONFIGURATION DES SYSTÈMES DYNAMIQUES
# ==========================================
# 3 foyers disposés en triangle
centers = [-1-1j, 1-1j, 0+1j]
weights = [1/3, 1/3, 1/3]
P_initial = 2 + 2j

# --- SCÉNARIO 1 : RÉGIME CONTRACTANT (|λ| < 1) ---
# En introduisant des phases initiales (phi_i) différentes, 
# la somme des phaseurs s'annule partiellement, induisant une homothétie globale < 1.
phases_contract = [0.0, np.pi/2, np.pi]
engine_contract = tth_core.TTH_Elementary(centers, weights, phases_contract)
theta_contract = np.pi / 6

lambda_contract = engine_contract.get_eigenvalue(theta_contract)
P_infty = engine_contract.get_fixed_point(theta_contract)
orbit_contract = generate_orbit(engine_contract, P_initial, theta_contract, 50)

# --- SCÉNARIO 2 : RÉGIME PÉRIODIQUE (|λ| = 1, ρ = p/q) ---
# Phases nulles = isométrie parfaite.
phases_iso = [0.0, 0.0, 0.0]
engine_iso = tth_core.TTH_Elementary(centers, weights, phases_iso)

# Angle rationnel vis-à-vis de 2π : theta = 2π * (1/5) => Période q = 5
theta_periodic = 2 * np.pi * (1/5)
lambda_periodic = engine_iso.get_eigenvalue(theta_periodic)
orbit_periodic = generate_orbit(engine_iso, P_initial, theta_periodic, 20)

# --- SCÉNARIO 3 : RÉGIME QUASI-PÉRIODIQUE (|λ| = 1, ρ ∉ Q) ---
# Angle irrationnel vis-à-vis de 2π : theta = 2π * (1 / racine(2)) => Dense
theta_quasi = 2 * np.pi * (1 / np.sqrt(2))
lambda_quasi = engine_iso.get_eigenvalue(theta_quasi)
orbit_quasi = generate_orbit(engine_iso, P_initial, theta_quasi, 1000)

# ==========================================
# VISUALISATION DES SPECTRES
# ==========================================
plt.figure(figsize=(18, 6))

def plot_complex_pts(ax, points, color, marker, label, line=False):
    x = [p.real for p in points]
    y = [p.imag for p in points]
    if line:
        ax.plot(x, y, color=color, alpha=0.5)
    ax.scatter(x, y, color=color, marker=marker, label=label)

# 1. Attracteur de Banach
ax1 = plt.subplot(131)
ax1.set_title(f"Régime Contractant (|λ| = {abs(lambda_contract):.2f})", fontweight='bold')
plot_complex_pts(ax1, orbit_contract, 'purple', '.', 'Orbite', line=True)
ax1.scatter([P_infty.real], [P_infty.imag], c='gold', marker='*', s=300, edgecolors='black', label='P_∞ (Banach)')
ax1.grid(True, linestyle=':')
ax1.legend()
ax1.set_aspect('equal')

# 2. Orbite Périodique
ax2 = plt.subplot(132)
ax2.set_title(f"Orbite Périodique (ρ = 1/5)", fontweight='bold')
plot_complex_pts(ax2, orbit_periodic, 'blue', 'o', 'Orbite (q=5)')
# Trace les centres
plot_complex_pts(ax2, centers, 'red', 'x', 'Foyers O_i')
ax2.grid(True, linestyle=':')
ax2.legend()
ax2.set_aspect('equal')

# 3. Orbite Quasi-Périodique
ax3 = plt.subplot(133)
ax3.set_title(f"Orbite Quasi-Périodique (ρ = 1/√2)", fontweight='bold')
plot_complex_pts(ax3, orbit_quasi, 'green', ',', 'Orbite Dense')
plot_complex_pts(ax3, centers, 'red', 'x', 'Foyers O_i')
ax3.grid(True, linestyle=':')
ax3.legend()
ax3.set_aspect('equal')

plt.tight_layout()
plt.show()

print("--- Analyse Spectrale TTH ---")
print(f"Contractant      : |λ| = {abs(lambda_contract):.4f} | Limite P_∞ = {P_infty:.2f}")
print(f"Périodique       : |λ| = {abs(lambda_periodic):.4f} | Angle ρ = 1/5 (Rationnel)")
print(f"Quasi-Périodique : |λ| = {abs(lambda_quasi):.4f} | Angle ρ = 1/√2 (Irrationnel)")
