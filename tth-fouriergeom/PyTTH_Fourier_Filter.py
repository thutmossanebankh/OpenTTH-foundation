

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. GÉNÉRATION DE LA FIGURE SPATIALE
# ==========================================
# On simule un polygone complexe de n=300 sommets (ex: contour d'une cellule en biologie)
n_vertices = 300
t = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)

# Forme géométrique de base (un cercle déformé en ellipse)
x_base = 1.5 * np.cos(t)
y_base = 1.0 * np.sin(t)

# Ajout d'un "bruit géométrique" (hautes fréquences et aléatoire)
noise_x = 0.08 * np.random.randn(n_vertices) + 0.1 * np.cos(25 * t)
noise_y = 0.08 * np.random.randn(n_vertices) + 0.1 * np.sin(30 * t)

# Représentation de la figure dans le plan complexe
polygon_noisy = (x_base + noise_x) + 1j * (y_base + noise_y)

# ==========================================
# 2. ISOMORPHISME DE FOURIER (TTH Harmonique)
# ==========================================
# L'analyse spectrale géométrique extrait les coefficients c_k de la TTH
# Chaque coefficient représente un "mode de déformation" du polygone
c_k = np.fft.fft(polygon_noisy)

# ==========================================
# 3. FILTRAGE SPATIAL (Lissage de la Courbure)
# ==========================================
# On agit sur l'espace réciproque : on annule l'influence des foyers à haute fréquence
# (Ceux qui causent les "tremblements" locaux)
modes_to_keep = 4 # On ne garde que la position (0) et les 3 premières harmoniques (forme globale)

c_k_filtered = np.copy(c_k)
# Mise à zéro (annulation) de tous les modes supérieurs
c_k_filtered[modes_to_keep : -modes_to_keep+1] = 0 

# ==========================================
# 4. RECONSTRUCTION GÉOMÉTRIQUE
# ==========================================
# On repasse dans l'espace euclidien avec les poids modifiés
polygon_smooth = np.fft.ifft(c_k_filtered)

# ==========================================
# 5. VISUALISATION DES RÉSULTATS
# ==========================================
# Pour fermer les polygones sur le graphique
def close_poly(p): return np.append(p, p[0])

plt.figure(figsize=(14, 7))

# Figure 1 : L'espace d'acquisition initial
ax1 = plt.subplot(121)
ax1.set_title("Contour Spatial Bruité (n=300 sommets)", fontweight='bold')
ax1.plot(close_poly(polygon_noisy).real, close_poly(polygon_noisy).imag, color='red', alpha=0.7, label='Figure Initiale')
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':')
ax1.legend(loc='upper right')

# Figure 2 : L'espace reconstruit par la TTH
ax2 = plt.subplot(122)
ax2.set_title(f"Reconstruction TTH Harmonique ({modes_to_keep} modes)", fontweight='bold')
ax2.plot(close_poly(polygon_noisy).real, close_poly(polygon_noisy).imag, color='lightgray', alpha=0.5, label='Bruit initial')
ax2.plot(close_poly(polygon_smooth).real, close_poly(polygon_smooth).imag, color='blue', linewidth=3, label='Forme Fondamentale Lissée')
ax2.set_aspect('equal')
ax2.grid(True, linestyle=':')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("--- Isomorphisme de Fourier Géométrique ---")
print(f"Nombre total de sommets        : {n_vertices}")
print(f"Modes de déformation conservés : {modes_to_keep}")
print("Compression spatiale réussie : L'information globale est préservée malgré l'annulation des hautes fréquences.")