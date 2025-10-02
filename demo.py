"""
Démonstration du framework Sitiai
"""

import numpy as np
import sitiai

print("=" * 60)
print("Bienvenue dans Sitiai!")
print("Framework Python pour créer et entraîner des IA légères")
print("=" * 60)

# ============================================
# DEMO 1: IA GÉNÉRATIVE - Générateur de noms
# ============================================

print("\n\n📝 DÉMO 1: Générateur de Noms")
print("-" * 60)

# Créer une IA générative pour les noms
print("\n1. Création d'une IA générative...")
name_ai = sitiai.create.ai('generative', mode='name_generator')
print(f"✓ {name_ai}")

# Charger des données d'entraînement
print("\n2. Chargement des données d'entraînement...")
noms_francais = [
    "Clemylia", "Alexandre", "Sophie", "Marie", "Pierre",
    "Julien", "Camille", "Lucas", "Emma", "Hugo",
    "Léa", "Thomas", "Chloé", "Nathan", "Manon",
    "Baptiste", "Clara", "Antoine", "Juliette", "Maxime"
]
name_ai.load_data(noms_francais)
print(f"✓ {len(noms_francais)} noms chargés")

# Entraîner le modèle
print("\n3. Entraînement du modèle...")
name_ai.train(epochs=100)
print("✓ Entraînement terminé!")

# Générer de nouveaux noms
print("\n4. Génération de nouveaux noms:")
nouveaux_noms = name_ai.generate_batch(n=8, temperature=0.8)
for i, nom in enumerate(nouveaux_noms, 1):
    print(f"   {i}. {nom}")

# ============================================
# DEMO 2: IA DE PRÉDICTION LINÉAIRE
# ============================================

print("\n\n" + "=" * 60)
print("📊 DÉMO 2: Prédiction Linéaire")
print("-" * 60)

# Créer des données synthétiques pour la régression
print("\n1. Création de données synthétiques...")
np.random.seed(42)
n_samples = 200

# Relation: y = 3*x1 + 2*x2 - x3 + 5 + bruit
X_train = np.random.randn(n_samples, 3)
y_train = 3 * X_train[:, 0] + 2 * X_train[:, 1] - X_train[:, 2] + 5
y_train += np.random.randn(n_samples) * 0.5  # Ajouter du bruit

X_test = np.random.randn(50, 3)
y_test = 3 * X_test[:, 0] + 2 * X_test[:, 1] - X_test[:, 2] + 5
y_test += np.random.randn(50) * 0.5

print(f"✓ {n_samples} échantillons d'entraînement, 50 échantillons de test")

# Créer une IA de prédiction linéaire
print("\n2. Création d'une IA de prédiction linéaire...")
linear_ai = sitiai.create.ai('linear', input_size=3, output_size=1, hidden_layers=[16, 8])
print(f"✓ {linear_ai}")

# Entraîner le modèle
print("\n3. Entraînement du modèle (cela peut prendre quelques secondes)...")
linear_ai.train(X_train, y_train, epochs=200, learning_rate=0.01, verbose=False)
print("✓ Entraînement terminé!")

# Évaluer le modèle
print("\n4. Évaluation sur les données de test...")
mse, r2 = linear_ai.evaluate(X_test, y_test)
print(f"   • Erreur quadratique moyenne (MSE): {mse:.4f}")
print(f"   • Score R² (coefficient de détermination): {r2:.4f}")

# Faire quelques prédictions
print("\n5. Exemples de prédictions:")
for i in range(5):
    x = X_test[i:i+1]
    pred = linear_ai.predict(x)[0, 0]
    real = y_test[i]
    print(f"   Prédiction: {pred:.2f} | Réel: {real:.2f} | Erreur: {abs(pred - real):.2f}")

# ============================================
# DEMO 3: Utilisation de SitiNEUR directement
# ============================================

print("\n\n" + "=" * 60)
print("🧠 DÉMO 3: Utilisation directe de SitiNEUR")
print("-" * 60)

print("\n1. Création d'une couche de neurones...")
layer = sitiai.SitiNEUR(input_size=5, output_size=3, activation='relu')
print(f"✓ {layer}")

print("\n2. Test de propagation avant...")
input_data = np.random.randn(2, 5)  # 2 exemples, 5 features
output = layer.forward(input_data)
print(f"   Entrée shape: {input_data.shape}")
print(f"   Sortie shape: {output.shape}")
print(f"   Sortie:\n{output}")

# ============================================
# DEMO 4: Sauvegarde et Chargement de Modèles
# ============================================

print("\n\n" + "=" * 60)
print("💾 DÉMO 4: Sauvegarde et Chargement de Modèles")
print("-" * 60)

print("\n1. Sauvegarde du modèle linéaire...")
linear_ai.save_weights('demo_linear_model.npz')

print("\n2. Sauvegarde du générateur de noms...")
name_ai.save_weights('demo_name_generator.npz')

print("\n3. Test de chargement...")
loaded_ai = sitiai.create.ai('linear', input_size=3, output_size=1, hidden_layers=[16, 8])
loaded_ai.load_weights('demo_linear_model.npz')

# Vérifier que ça fonctionne
test_pred = loaded_ai.predict(X_test[0:1])
print(f"   Prédiction avec modèle chargé: {test_pred[0, 0]:.2f}")

# ============================================
# Conclusion
# ============================================

print("\n\n" + "=" * 60)
print("✨ Démonstration terminée!")
print("=" * 60)
print("\nSitiai vous permet de:")
print("  • Créer des IA génératives pour générer des noms, textes, etc.")
print("  • Créer des IA de prédiction linéaire pour la régression")
print("  • Utiliser des couches de neurones SitiNEUR facilement")
print("  • Sauvegarder et charger vos modèles (.npz)")
print("  • Partager vos modèles facilement!")
print("\nSyntaxe simple inspirée de PyTorch, mais plus accessible!")
print("=" * 60)
