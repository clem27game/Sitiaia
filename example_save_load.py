
"""
Exemple de sauvegarde et chargement de modèles Sitiai
"""

import numpy as np
import sitiai

print("=" * 60)
print("Exemple: Sauvegarde et Chargement de Modèles")
print("=" * 60)

# ============================================
# EXEMPLE 1: Modèle Linéaire
# ============================================

print("\n📊 EXEMPLE 1: Modèle de Régression Linéaire")
print("-" * 60)

# Créer des données
np.random.seed(42)
X_train = np.random.randn(100, 3)
y_train = 3 * X_train[:, 0] + 2 * X_train[:, 1] - X_train[:, 2] + 5 + np.random.randn(100) * 0.5

# Créer et entraîner le modèle
print("\n1. Création et entraînement du modèle...")
ai = sitiai.create.ai('linear', input_size=3, output_size=1, hidden_layers=[16, 8])
ai.train(X_train, y_train, epochs=100, verbose=False)
print(f"✓ Modèle entraîné: {ai}")

# Faire une prédiction avant sauvegarde
X_test = np.array([[1.0, 2.0, 0.5]])
pred_before = ai.predict(X_test)
print(f"\n2. Prédiction avant sauvegarde: {pred_before[0, 0]:.4f}")

# Sauvegarder le modèle
print("\n3. Sauvegarde du modèle...")
ai.save_weights('my_linear_model.npz')

# Créer un nouveau modèle et charger les poids
print("\n4. Création d'un nouveau modèle et chargement des poids...")
ai_loaded = sitiai.create.ai('linear', input_size=3, output_size=1, hidden_layers=[16, 8])
ai_loaded.load_weights('my_linear_model.npz')

# Vérifier que les prédictions sont identiques
pred_after = ai_loaded.predict(X_test)
print(f"\n5. Prédiction après chargement: {pred_after[0, 0]:.4f}")
print(f"   Différence: {abs(pred_before[0, 0] - pred_after[0, 0]):.10f}")

# ============================================
# EXEMPLE 2: Modèle Génératif
# ============================================

print("\n\n" + "=" * 60)
print("📝 EXEMPLE 2: Générateur de Noms")
print("-" * 60)

# Créer et entraîner le modèle
print("\n1. Création et entraînement du modèle génératif...")
name_ai = sitiai.create.ai('generative', mode='name_generator')
noms = ["Alexandre", "Sophie", "Marie", "Pierre", "Julien", "Camille"]
name_ai.load_data(noms)
name_ai.train(epochs=100)
print(f"✓ {name_ai}")

# Générer avant sauvegarde
print("\n2. Génération avant sauvegarde:")
names_before = name_ai.generate_batch(n=3, temperature=0.8)
for i, nom in enumerate(names_before, 1):
    print(f"   {i}. {nom}")

# Sauvegarder
print("\n3. Sauvegarde du modèle...")
name_ai.save_weights('name_generator.npz')

# Charger dans un nouveau modèle
print("\n4. Chargement dans un nouveau modèle...")
name_ai_loaded = sitiai.create.ai('generative')
name_ai_loaded.load_weights('name_generator.npz')

# Générer après chargement
print("\n5. Génération après chargement:")
names_after = name_ai_loaded.generate_batch(n=3, temperature=0.8)
for i, nom in enumerate(names_after, 1):
    print(f"   {i}. {nom}")

# ============================================
# EXEMPLE 3: Partage de Modèle
# ============================================

print("\n\n" + "=" * 60)
print("🌐 EXEMPLE 3: Partage de Modèle")
print("-" * 60)

print("""
Vos modèles peuvent maintenant être partagés facilement!

Exemples d'utilisation:
1. Sauvegarder localement:
   ai.save_weights('mon_modele.npz')

2. Partager sur GitHub:
   - Commitez le fichier .npz dans votre repo
   - Les utilisateurs peuvent télécharger et charger avec load_weights()

3. Publier sur un service cloud:
   - Upload le fichier .npz
   - Partagez le lien de téléchargement

4. Intégrer dans une application:
   - Incluez le fichier .npz dans votre package
   - Chargez-le au démarrage de l'application

Le format .npz est compact et compatible avec NumPy!
""")

print("=" * 60)
print("✨ Démonstration terminée!")
print("=" * 60)
