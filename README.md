
# 🧠 Sitiai

> **Framework Python léger pour créer et entraîner des IA simples**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.3%2B-orange.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-red.svg)](pyproject.toml)

---

## 📖 Table des matières

- [🚀 Installation](#-installation)
- [✨ Fonctionnalités](#-fonctionnalités)
- [🤖 IA Génératives](#-ia-génératives)
- [📊 IA de Prédiction Linéaire](#-ia-de-prédiction-linéaire)
- [🧬 Couches de Neurones SitiNEUR](#-couches-de-neurones-sitinetur)
- [📚 Exemples Complets](#-exemples-complets)
- [⚙️ Configuration Avancée](#️-configuration-avancée)
- [🔧 API Reference](#-api-reference)
- [📝 Licence](#-licence)

---

## 🚀 Installation

### Option 1: Depuis GitHub (Recommandé)

```bash
# Cloner le repository
git clone https://github.com/clem27game/Sitiaia.git
cd Sitiaia

# Installer les dépendances
pip install numpy>=2.3.3

# Installer le package en mode développement
pip install -e .
```

### Option 2: Installation directe

```bash
pip install git+https://github.com/clem27game/Sitiaia.git
```

### Vérification de l'installation

```python
import Sitiaia
print(f"Sitiai version: {sitiaia.__version__}")
# Output: Sitiaia version: 0.1.0
```

---

## ✨ Fonctionnalités

| Fonctionnalité | Description | Status |
|----------------|-------------|--------|
| 🎨 **IA Génératives** | Génération de noms, textes, contenus | ✅ |
| 📈 **IA de Prédiction** | Régression linéaire et non-linéaire | ✅ |
| 🧠 **Couches Neuronales** | API simple inspirée de PyTorch | ✅ |
| 🔥 **Activations** | ReLU, Sigmoid, Tanh, Linear | ✅ |
| 📦 **Léger** | Seulement NumPy comme dépendance | ✅ |
| 🎓 **Éducatif** | Code clair et accessible | ✅ |

---

## 🤖 IA Génératives

### 🏷️ Générateur de Noms

Créez des IA capables de générer des noms originaux à partir d'exemples :

```python
import sitiai

# 1. Créer une IA générative
ai = sitiai.create.ai('generative', mode='name_generator')

# 2. Charger vos données d'exemples
noms_francais = [
    "Alexandre", "Sophie", "Marie", "Pierre", "Julien",
    "Camille", "Lucas", "Emma", "Hugo", "Léa"
]
ai.load_data(noms_francais)

# 3. Entraîner le modèle
ai.train(epochs=100)

# 4. Générer de nouveaux noms
nouveau_nom = ai.generate()
print(f"Nouveau nom: {nouveau_nom}")

# 5. Générer plusieurs noms avec créativité
noms_creatifs = ai.generate_batch(n=5, temperature=0.8)
for nom in noms_creatifs:
    print(f"✨ {nom}")
```

### 🎛️ Contrôler la créativité

```python
# Faible créativité (plus proche des exemples)
noms_conservateurs = ai.generate_batch(n=3, temperature=0.3)

# Haute créativité (plus original)
noms_originaux = ai.generate_batch(n=3, temperature=1.5)
```

---

## 📊 IA de Prédiction Linéaire

### 📈 Régression Simple

Créez des modèles de prédiction pour vos données numériques :

```python
import sitiai
import numpy as np

# 1. Préparer vos données
# Exemple: prédire le prix d'une maison selon surface, chambres, âge
X_train = np.array([
    [100, 3, 5],   # 100m², 3 chambres, 5 ans
    [80, 2, 10],   # 80m², 2 chambres, 10 ans
    [120, 4, 2],   # 120m², 4 chambres, 2 ans
    # ... plus de données
])
y_train = np.array([250000, 180000, 320000])  # Prix en euros

# 2. Créer une IA de prédiction
ai = sitiai.create.ai(
    'linear', 
    input_size=3,      # 3 caractéristiques
    output_size=1,     # 1 prédiction (prix)
    hidden_layers=[16, 8]  # Couches cachées
)

# 3. Entraîner le modèle
ai.train(X_train, y_train, epochs=200, learning_rate=0.01)

# 4. Faire des prédictions
nouvelle_maison = np.array([[90, 2, 7]])  # 90m², 2 chambres, 7 ans
prix_predit = ai.predict(nouvelle_maison)
print(f"Prix prédit: {prix_predit[0, 0]:.0f}€")
```

### 📊 Évaluation du modèle

```python
# Évaluer sur des données de test
X_test = np.array([[110, 3, 3], [75, 2, 15]])
y_test = np.array([280000, 150000])

mse, r2 = ai.evaluate(X_test, y_test)
print(f"Erreur quadratique: {mse:.2f}")
print(f"Score R²: {r2:.3f}")
```

---

## 🧬 Couches de Neurones SitiNEUR

### 🔗 Utilisation Directe

Pour un contrôle fin, utilisez directement les couches neuronales :

```python
from sitiai import SitiNEUR
import numpy as np

# Créer une couche de neurones
layer = SitiNEUR(
    input_size=10, 
    output_size=5, 
    activation='relu'
)

# Données d'entrée (batch de 3 exemples)
input_data = np.random.randn(3, 10)

# Propagation avant
output = layer.forward(input_data)
print(f"Forme d'entrée: {input_data.shape}")
print(f"Forme de sortie: {output.shape}")
```

### 🔄 Entraînement Manuel

```python
# Simulation d'un gradient
grad_output = np.random.randn(3, 5)

# Rétropropagation
grad_input = layer.backward(grad_output, learning_rate=0.01)
print(f"Gradient d'entrée: {grad_input.shape}")
```

---

## 📚 Exemples Complets

### 🎯 Exemple 1: Prédiction de Température

```python
import sitiai
import numpy as np

# Données: [humidité, pression, vent] -> température
X = np.random.randn(1000, 3)
y = 20 + 2*X[:, 0] - 0.5*X[:, 1] + 0.1*X[:, 2] + np.random.randn(1000)*2

# Séparer train/test
split = 800
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Créer et entraîner le modèle
ai = sitiai.create.ai('linear', input_size=3, output_size=1)
ai.train(X_train, y_train, epochs=150, verbose=True)

# Évaluer
mse, r2 = ai.evaluate(X_test, y_test)
print(f"🌡️ Précision du modèle météo: R² = {r2:.3f}")
```

### 🏷️ Exemple 2: Générateur de Marques

```python
import sitiai

# Noms de marques technologiques
marques_tech = [
    "Google", "Apple", "Microsoft", "Amazon", "Meta",
    "Tesla", "Netflix", "Spotify", "Adobe", "Oracle"
]

# Créer l'IA
brand_ai = sitiai.create.ai('generative', mode='name_generator')
brand_ai.load_data(marques_tech)
brand_ai.train(epochs=150)

# Générer de nouvelles marques
print("🚀 Nouvelles marques générées:")
for i, marque in enumerate(brand_ai.generate_batch(n=5), 1):
    print(f"   {i}. {marque}")
```

---

## ⚙️ Configuration Avancée

### 🔧 Paramètres d'Entraînement

```python
# Configuration fine pour LinearAI
ai = sitiai.create.ai('linear', input_size=5, output_size=1, hidden_layers=[32, 16, 8])

ai.train(
    X_train, y_train,
    epochs=300,           # Nombre d'époques
    learning_rate=0.001,  # Taux d'apprentissage
    batch_size=32,        # Taille des mini-batches
    verbose=True          # Affichage des logs
)
```

### 🎨 Paramètres Génératifs

```python
# Configuration pour GenerativeAI
ai = sitiai.create.ai('generative', mode='name_generator')
ai.load_data(data)
ai.train(epochs=200, ngram_size=3)  # Trigrammes au lieu de bigrammes

# Génération avec contrôle
result = ai.generate(
    max_length=15,      # Longueur maximale
    temperature=1.2     # Créativité
)
```

---

## 🔧 API Reference

### 📋 Fonctions d'Activation Supportées

| Activation | Formule | Usage |
|------------|---------|--------|
| `'relu'` | `max(0, x)` | Couches cachées (défaut) |
| `'sigmoid'` | `1/(1+e^(-x))` | Classification binaire |
| `'tanh'` | `tanh(x)` | Données centrées |
| `'linear'` | `x` | Couche de sortie |

### 🏗️ Architecture des Modèles

```python
# LinearAI avec architecture personnalisée
ai = sitiai.create.ai(
    'linear',
    input_size=10,           # Taille d'entrée
    output_size=1,           # Taille de sortie
    hidden_layers=[64, 32, 16]  # Couches cachées
)

# Structure résultante: 10 -> 64 -> 32 -> 16 -> 1
print(ai)  # Affiche l'architecture
```

### 📊 Métriques d'Évaluation

```python
# Obtenir les métriques détaillées
predictions = ai.predict(X_test)
mse, r2 = ai.evaluate(X_test, y_test)

# Calculs manuels
mae = np.mean(np.abs(predictions - y_test))  # Erreur absolue moyenne
print(f"MAE: {mae:.4f}")
```

---

## 🎓 Utilisation Éducative

Sitiai est parfait pour :

- 📚 **Apprentissage** des concepts de ML
- 🔬 **Prototypage** rapide d'idées
- 🎯 **Projets étudiants** en IA
- 🚀 **Applications légères** sans complexité

### 💡 Exemple pour Débutants

```python
# Super simple: prédire y = 2x + 1
import sitiai
import numpy as np

# Données simples
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

# Créer et entraîner
ai = sitiai.create.ai('linear', input_size=1, output_size=1)
ai.train(X, y, epochs=100)

# Tester
test_x = np.array([[6]])
prediction = ai.predict(test_x)
print(f"Pour x=6, y prédit = {prediction[0,0]:.1f}")  # Devrait être ~13
```

---

## 🤝 Contribution

Envie de contribuer ? Voici comment :

1. 🍴 Fork le repository
2. 🌿 Créez une branche pour votre feature
3. ✏️ Commitez vos changements
4. 📤 Push vers la branche
5. 🔄 Ouvrez une Pull Request

---

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 👨‍💻 Auteur

**Clemylia** - Créateur de Sitiai

---

<div align="center">

**⭐ N'oubliez pas de donner une étoile si Sitiai vous aide ! ⭐**

[🐛 Reporter un Bug](https://github.com/clemylia/sitiai/issues) | [💡 Demander une Feature](https://github.com/clem27game/sitiaia/issues) | [📖 Documentation](https://github.com/clem27game/sitiaia)

</div>

🛑 **Attention** : Remplacez sitiai par Sitiaia dans les codes pour que la version obtenu avec le repos github fonctionne