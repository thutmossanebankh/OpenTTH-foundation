import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ==========================================
# 1. DÉFINITION DE LA COUCHE DE THUTMOS (TTH)
# ==========================================
class TTHLayer(nn.Module):
    def __init__(self, n_centers=2):
        super(TTHLayer, self).__init__()
        self.n_centers = n_centers
        
        # Paramètres géométriques APPRENABLES (Les "Poids" de notre couche)
        # O_i : Coordonnées des foyers d'action (initialisés aléatoirement)
        self.centers = nn.Parameter(torch.randn(n_centers, 2))
        
        # theta_i : Angles de balayage spectraux (initialisés proches de 0)
        self.angles = nn.Parameter(torch.randn(n_centers) * 0.1)
        
        # Optionnel : Un paramètre de "température" pour contrôler la portée des Gaussiennes
        self.gamma = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        """
        x: Tenseur de forme [batch_size, 2]
        Retourne l'espace transformé [batch_size, 2]
        """
        batch_size = x.size(0)
        transformed_x = torch.zeros_like(x)
        
        # Calcul des poids spatiaux lambda_i(x) pour chaque point et chaque foyer
        # Forme : [batch_size, n_centers]
        distances_sq = torch.cdist(x, self.centers) ** 2
        # On utilise Softmax pour garantir la partition de l'unité (Somme des lambda = 1)
        weights = torch.softmax(-self.gamma * distances_sq, dim=1)
        
        # Pour chaque foyer de Thutmos
        for i in range(self.n_centers):
            # Matrice de rotation R_i
            theta = self.angles[i]
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            R = torch.stack([
                torch.stack([cos_t, -sin_t]),
                torch.stack([sin_t,  cos_t])
            ])
            
            # Vecteur P - O_i
            diff = x - self.centers[i]
            
            # Application de la rotation: R_i(P - O_i)
            # diff est [batch, 2], R est [2, 2]. Multiplication matricielle batched.
            rotated = torch.matmul(diff, R.t())
            
            # Translation de retour: R_i(P - O_i) + O_i
            F_i = rotated + self.centers[i]
            
            # Combinaison convexe pondérée par lambda_i(x)
            w_i = weights[:, i].unsqueeze(1) # [batch, 1]
            transformed_x += w_i * F_i
            
        return transformed_x

# Modèle complet : TTH géométrique + Classifieur Linéaire
class TTH_Classifier(nn.Module):
    def __init__(self, n_centers=2):
        super().__init__()
        self.geom_layer = TTHLayer(n_centers=n_centers)
        self.linear = nn.Linear(2, 1) # Trace une ligne droite dans l'espace déformé
        
    def forward(self, x):
        h = self.geom_layer(x)
        out = torch.sigmoid(self.linear(h))
        return out, h # On retourne aussi 'h' pour visualiser l'espace déformé

# ==========================================
# 2. ENTRAÎNEMENT DU MODÈLE (BOUCLE D'OPTIMISATION)
# ==========================================
# Génération des données "Moons" (non-linéaires)
X_np, y_np = make_moons(n_samples=300, noise=0.1, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

# Initialisation
model = TTH_Classifier(n_centers=4) # Mettons 4 foyers pour lui donner de la souplesse !
optimizer = optim.Adam(model.parameters(), lr=0.05) # Adam va chercher les meilleurs foyers
criterion = nn.BCELoss() # Binary Cross Entropy

print("--- Début de l'entraînement géométrique ---")
epochs = 300
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions, latent_space = model(X)
    loss = criterion(predictions, y)
    loss.backward()  # La magie PyTorch : calcule les gradients de vos matrices Jacobiennes
    optimizer.step() # Déplace les foyers et ajuste les angles
    
    loss_history.append(loss.item())
    
    if (epoch+1) % 50 == 0:
        # Calcul de la précision
        acc = ((predictions > 0.5).float() == y).float().mean().item()
        print(f"Époque {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f} | Précision: {acc*100:.1f}%")

# ==========================================
# 3. VISUALISATION DES RÉSULTATS
# ==========================================
# On désactive le calcul des gradients pour l'inférence
model.eval()
with torch.no_grad():
    _, X_transformed = model(X)
    X_trans_np = X_transformed.numpy()
    centers_np = model.geom_layer.centers.numpy()

plt.figure(figsize=(15, 5))

# Plot 1: Courbe de Loss
plt.subplot(131)
plt.plot(loss_history, color='purple', lw=2)
plt.title("Convergence de l'Erreur")
plt.xlabel("Époques")
plt.ylabel("BCE Loss")
plt.grid(True, linestyle=':')

# Plot 2: Espace Original
plt.subplot(132)
plt.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], c='red', alpha=0.6)
plt.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], c='blue', alpha=0.6)
plt.title("Espace Euclidien Original")
plt.grid(True, linestyle=':')

# Plot 3: Espace Déformé par TTH
plt.subplot(133)
plt.scatter(X_trans_np[y_np==0, 0], X_trans_np[y_np==0, 1], c='red', alpha=0.6)
plt.scatter(X_trans_np[y_np==1, 0], X_trans_np[y_np==1, 1], c='blue', alpha=0.6)
plt.scatter(centers_np[:, 0], centers_np[:, 1], c='green', marker='x', s=100, linewidths=3, label='Foyers Appris')
plt.title("Espace Riemannien Appris (Déplié)")
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()

print("\n--- Paramètres Géométriques Appris ---")
print("Coordonnées des Foyers (O_i) :\n", centers_np)
print("Angles spectraux (theta_i) :\n", model.geom_layer.angles.detach().numpy())

