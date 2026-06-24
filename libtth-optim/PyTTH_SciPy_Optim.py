import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import tth_diff

# 1. Génération des données
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# 2. Fonction Objectif (Ce que SciPy doit minimiser)
def objective_function(params):
    # Décryptage des paramètres envoyés par SciPy (pour k=2 centres)
    # params = [cx1, cy1, cx2, cy2, theta1, theta2]
    c1 = [params[0], params[1]]
    c2 = [params[2], params[3]]
    centers = [c1, c2]
    angles = [params[4], params[5]]
    
    # Définition dynamique des champs scalaires pour cette itération
    def l1(x): return np.exp(-((x[0]-c1[0])**2 + (x[1]-c1[1])**2))
    def l2(x): return np.exp(-((x[0]-c2[0])**2 + (x[1]-c2[1])**2))
    
    def norm_l1(x): return l1(x) / (l1(x) + l2(x) + 1e-9)
    def norm_l2(x): return l2(x) / (l1(x) + l2(x) + 1e-9)
    
    # Pour l'optimisation sans gradient (Nelder-Mead), nous pouvons fournir des
    # gradients factices car tth_diff a besoin de ces fonctions à l'initialisation,
    # mais l'évaluation simple evaluate(x) n'utilise pas le gradient.
    def dummy_grad(x): return [0.0, 0.0]
    
    # Initialisation du moteur C++
    engine = tth_diff.TTH_Differential(centers, angles, [norm_l1, norm_l2], [dummy_grad, dummy_grad])
    
    # Transformation de l'espace
    X_trans = np.zeros_like(X)
    for i in range(X.shape[0]):
        pt = engine.evaluate(X[i])
        X_trans[i, 0] = pt[0]
        X_trans[i, 1] = pt[1]
        
    # Évaluation de la séparation avec une Régression Logistique très rapide
    clf = LogisticRegression(max_iter=100)
    clf.fit(X_trans, y)
    probs = clf.predict_proba(X_trans)
    
    # On retourne la Log Loss (Erreur continue, plus facile à optimiser que l'Accuracy)
    return log_loss(y, probs)

# 3. Lancement de l'Optimisation Numérique
print("Lancement de l'optimiseur SciPy (Nelder-Mead) sur le moteur C++...")
print("Recherche de la topologie géométrique optimale (cela peut prendre quelques secondes)...")

# Paramètres initiaux (Pire scénario : au centre avec angles nuls)
initial_guess = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

# Optimisation sans gradient mathématique (black-box optimisation)
#result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': 250, 'disp': True})

result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': 2000, 'disp': True})


print("\n--- Résultats de l'Optimisation ---")
print("Succès :", result.success)
print("Log Loss Finale :", result.fun)

# 4. Évaluation finale avec les meilleurs paramètres
best_p = result.x
best_c1, best_c2 = [best_p[0], best_p[1]], [best_p[2], best_p[3]]
best_angles = [best_p[4], best_p[5]]

def l1(x): return np.exp(-((x[0]-best_c1[0])**2 + (x[1]-best_c1[1])**2))
def l2(x): return np.exp(-((x[0]-best_c2[0])**2 + (x[1]-best_c2[1])**2))
def norm_l1(x): return l1(x) / (l1(x) + l2(x) + 1e-9)
def norm_l2(x): return l2(x) / (l1(x) + l2(x) + 1e-9)
def dummy_grad(x): return [0.0, 0.0]

engine = tth_diff.TTH_Differential([best_c1, best_c2], best_angles, [norm_l1, norm_l2], [dummy_grad, dummy_grad])

X_final = np.zeros_like(X)
for i in range(X.shape[0]):
    pt = engine.evaluate(X[i])
    X_final[i, 0] = pt[0]
    X_final[i, 1] = pt[1]

clf = LogisticRegression()
clf.fit(X_final, y)
acc = accuracy_score(y, clf.predict(X_final))

print(f"\nPrécision Finale (Accuracy) : {acc * 100:.1f}%")
print(f"Foyer 1 optimal : {best_c1} (Angle : {best_angles[0]:.2f} rad)")
print(f"Foyer 2 optimal : {best_c2} (Angle : {best_angles[1]:.2f} rad)")

