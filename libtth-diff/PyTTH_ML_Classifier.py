
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tth_diff

# --- 1. Génération du Jeu de Données Non-Linéaire ---
# make_moons génère deux demi-cercles entrelacés (très classique en ML)
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# --- 2. Configuration de la TTH (La "Couche Géométrique") ---
# Nous plaçons deux foyers (O1 et O2) stratégiquement près des centres des "lunes"
centers = [[0.0, 0.5], [1.0, -0.5]]
# Angles de rotation opposés pour "déplier" les lunes
#angles = [np.pi/2, -np.pi/2]


# On inverse le sens des rotations pour "ouvrir" l'espace au lieu de le fermer
angles = [-np.pi/2.5, np.pi/2.5] 



# Fonctions de poids basées sur la distance radiale (Gaussiennes)
# Plus on est proche d'un foyer, plus son influence est forte.
def lambda1(x):
    dist_sq = (x[0] - centers[0][0])**2 + (x[1] - centers[0][1])**2
    return np.exp(-dist_sq)

def lambda2(x):
    dist_sq = (x[0] - centers[1][0])**2 + (x[1] - centers[1][1])**2
    return np.exp(-dist_sq)

# Gradients analytiques des Gaussiennes
def grad_lambda1(x):
    w = lambda1(x)
    return [-2 * (x[0] - centers[0][0]) * w, -2 * (x[1] - centers[0][1]) * w]

def grad_lambda2(x):
    w = lambda2(x)
    return [-2 * (x[0] - centers[1][0]) * w, -2 * (x[1] - centers[1][1]) * w]

# Normalisation pour garantir la partition de l'unité (w1 + w2 = 1)
def norm_lambda1(x):
    s = lambda1(x) + lambda2(x)
    return lambda1(x) / s

def norm_lambda2(x):
    s = lambda1(x) + lambda2(x)
    return lambda2(x) / s

# Gradients normalisés (règle du quotient)
def norm_grad_lambda1(x):
    l1 = lambda1(x); l2 = lambda2(x); s = l1 + l2
    g1 = grad_lambda1(x); g2 = grad_lambda2(x)
    return [(g1[0]*s - l1*(g1[0]+g2[0])) / s**2, (g1[1]*s - l1*(g1[1]+g2[1])) / s**2]

def norm_grad_lambda2(x):
    l1 = lambda1(x); l2 = lambda2(x); s = l1 + l2
    g1 = grad_lambda1(x); g2 = grad_lambda2(x)
    return [(g2[0]*s - l2*(g1[0]+g2[0])) / s**2, (g2[1]*s - l2*(g1[1]+g2[1])) / s**2]

lambdas = [norm_lambda1, norm_lambda2]
grad_lambdas = [norm_grad_lambda1, norm_grad_lambda2]

# Initialisation du moteur C++
diff_engine = tth_diff.TTH_Differential(centers, angles, lambdas, grad_lambdas)

# --- 3. Déformation de l'Espace Latent ---
X_transformed = np.zeros_like(X)
for i in range(X.shape[0]):
    transformed_pt = diff_engine.evaluate(X[i])
    X_transformed[i, 0] = transformed_pt[0]
    X_transformed[i, 1] = transformed_pt[1]

# --- 4. Entraînement du Classifieur Linéaire (Régression Logistique) ---
# On entraîne un classifieur simple sur les données originales (qui va échouer)
clf_orig = LogisticRegression()
clf_orig.fit(X, y)
acc_orig = accuracy_score(y, clf_orig.predict(X))

# On entraîne le même classifieur sur les données transformées par la TTH
clf_tth = LogisticRegression()
clf_tth.fit(X_transformed, y)
acc_tth = accuracy_score(y, clf_tth.predict(X_transformed))

print(f"Précision sur l'espace original (Euclidien) : {acc_orig:.2f}")
print(f"Précision sur l'espace transformé (TTH) : {acc_tth:.2f}")

# --- 5. Visualisation des Résultats ---
plt.figure(figsize=(14, 6))

# Données Originales
plt.subplot(121)
plt.title(f"Espace Latent Original (Acc: {acc_orig:.2f})")
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', label='Classe 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', label='Classe 1', alpha=0.6)
# Tracé de la frontière de décision (qui est une ligne droite ici)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 50),
                     np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 50))
Z = clf_orig.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, colors=['red', 'blue'])
plt.scatter([c[0] for c in centers], [c[1] for c in centers], c='green', marker='x', s=100, label='Foyers TTH')
plt.legend()

# Données Transformées
plt.subplot(122)
plt.title(f"Espace Transformé par TTH (Acc: {acc_tth:.2f})")
plt.scatter(X_transformed[y==0, 0], X_transformed[y==0, 1], color='red', label='Classe 0', alpha=0.6)
plt.scatter(X_transformed[y==1, 0], X_transformed[y==1, 1], color='blue', label='Classe 1', alpha=0.6)
# Frontière de décision
xx_t, yy_t = np.meshgrid(np.linspace(X_transformed[:,0].min()-0.5, X_transformed[:,0].max()+0.5, 50),
                         np.linspace(X_transformed[:,1].min()-0.5, X_transformed[:,1].max()+0.5, 50))
Z_t = clf_tth.predict(np.c_[xx_t.ravel(), yy_t.ravel()]).reshape(xx_t.shape)
plt.contourf(xx_t, yy_t, Z_t, alpha=0.2, colors=['red', 'blue'])
plt.legend()

plt.tight_layout()
plt.show()

