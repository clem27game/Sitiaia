"""
Module pour les IA de prédiction linéaire
"""

import numpy as np
from typing import List, Tuple, Optional
from .neuron import SitiNEUR


class LinearAI:
    """
    IA de prédiction linéaire avec réseau de neurones
    
    Args:
        input_size: Nombre de features en entrée
        output_size: Nombre de sorties (1 pour régression simple)
        hidden_layers: Liste des tailles des couches cachées
        
    Example:
        >>> from sitiai import create
        >>> ai = create.ai('linear', input_size=3, output_size=1, hidden_layers=[10, 5])
        >>> ai.train(X_train, y_train, epochs=100)
        >>> predictions = ai.predict(X_test)
    """
    
    def __init__(self, input_size: int, output_size: int = 1, hidden_layers: Optional[List[int]] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers or [10]
        
        # Construire le réseau
        self.layers: List[SitiNEUR] = []
        
        # Première couche
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Utiliser 'relu' pour les couches cachées, 'linear' pour la sortie
            activation = 'relu' if i < len(layer_sizes) - 2 else 'linear'
            layer = SitiNEUR(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activation
            )
            self.layers.append(layer)
        
        self.is_trained = False
        self.loss_history = []
        
        # Paramètres de normalisation
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
    def _normalize_X(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalise les entrées"""
        if fit:
            self.x_mean = np.mean(X, axis=0)
            self.x_std = np.std(X, axis=0) + 1e-8  # Éviter division par zéro
        
        return (X - self.x_mean) / self.x_std
    
    def _normalize_y(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalise les sorties"""
        if fit:
            self.y_mean = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0) + 1e-8  # Éviter division par zéro
        
        return (y - self.y_mean) / self.y_std
    
    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        """Dénormalise les prédictions"""
        if self.y_mean is not None and self.y_std is not None:
            return y * self.y_std + self.y_mean
        return y
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propagation avant à travers tout le réseau
        
        Args:
            x: Données d'entrée
            
        Returns:
            Prédictions
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              learning_rate: float = 0.01, batch_size: Optional[int] = None,
              verbose: bool = True):
        """
        Entraîne le modèle sur les données
        
        Args:
            X: Données d'entrée (shape: [n_samples, input_size])
            y: Cibles (shape: [n_samples, output_size])
            epochs: Nombre d'époques d'entraînement
            learning_rate: Taux d'apprentissage
            batch_size: Taille des mini-batches (None = batch complet)
            verbose: Afficher les logs d'entraînement
        """
        # Convertir et vérifier les données
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # S'assurer que y est 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Vérifier qu'il n'y a pas de NaN dans les données
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Les données contiennent des valeurs NaN")
        
        # Normaliser les données
        X_norm = self._normalize_X(X, fit=True)
        y_norm = self._normalize_y(y, fit=True)
        
        n_samples = X_norm.shape[0]
        batch_size = batch_size or min(32, n_samples)
        
        self.loss_history = []
        
        for epoch in range(epochs):
            # Mélanger les données
            indices = np.random.permutation(n_samples)
            X_shuffled = X_norm[indices]
            y_shuffled = y_norm[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Entraînement par mini-batches
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                # Forward pass
                predictions = self.forward(batch_X)
                
                # Vérifier les NaN
                if np.any(np.isnan(predictions)):
                    if verbose:
                        print(f"⚠️ NaN détecté à l'époque {epoch + 1}, arrêt de l'entraînement")
                    break
                
                # Calculer la perte (MSE)
                loss = np.mean((predictions - batch_y) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                grad = 2 * (predictions - batch_y) / batch_y.shape[0]
                
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            
            if n_batches == 0:
                break
                
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Époque {epoch + 1}/{epochs}, Perte: {avg_loss:.6f}")
        
        self.is_trained = True
        
        if verbose:
            print(f"\n✓ Entraînement terminé! Perte finale: {self.loss_history[-1]:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données
        
        Args:
            X: Données d'entrée
            
        Returns:
            Prédictions
        """
        if not self.is_trained:
            print("⚠️ Attention: Le modèle n'est pas entraîné.")
        
        X = np.array(X, dtype=np.float64)
        
        # Normaliser les entrées si le modèle a été entraîné
        if self.x_mean is not None:
            X_norm = self._normalize_X(X, fit=False)
            predictions_norm = self.forward(X_norm)
            return self._denormalize_y(predictions_norm)
        
        return self.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Évalue le modèle sur des données de test
        
        Args:
            X: Données d'entrée
            y: Vraies valeurs
            
        Returns:
            (mse, r2_score)
        """
        predictions = self.predict(X)
        y = np.array(y, dtype=np.float32)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # MSE
        mse = np.mean((predictions - y) ** 2)
        
        # R² score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return mse, r2
    
    def save_weights(self, filepath: str):
        """
        Sauvegarde les poids du modèle dans un fichier
        
        Args:
            filepath: Chemin du fichier de sauvegarde (.npz)
            
        Example:
            >>> ai.save_weights('my_model.npz')
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        
        weights_data = {}
        
        # Sauvegarder les poids de chaque couche
        for i, layer in enumerate(self.layers):
            layer_weights = layer.get_weights()
            weights_data[f'layer_{i}_weights'] = layer_weights['weights']
            weights_data[f'layer_{i}_bias'] = layer_weights['bias']
        
        # Sauvegarder les paramètres de normalisation
        if self.x_mean is not None:
            weights_data['x_mean'] = self.x_mean
            weights_data['x_std'] = self.x_std
            weights_data['y_mean'] = self.y_mean
            weights_data['y_std'] = self.y_std
        
        # Sauvegarder la configuration
        weights_data['input_size'] = np.array([self.input_size])
        weights_data['output_size'] = np.array([self.output_size])
        weights_data['hidden_layers'] = np.array(self.hidden_layers)
        weights_data['is_trained'] = np.array([self.is_trained])
        
        np.savez(filepath, **weights_data)
        print(f"✓ Modèle sauvegardé dans '{filepath}'")
    
    def load_weights(self, filepath: str):
        """
        Charge les poids du modèle depuis un fichier
        
        Args:
            filepath: Chemin du fichier de sauvegarde (.npz)
            
        Example:
            >>> ai = sitiai.create.ai('linear', input_size=3, output_size=1)
            >>> ai.load_weights('my_model.npz')
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        
        data = np.load(filepath, allow_pickle=True)
        
        # Charger les poids de chaque couche
        for i, layer in enumerate(self.layers):
            weights_dict = {
                'weights': data[f'layer_{i}_weights'],
                'bias': data[f'layer_{i}_bias']
            }
            layer.set_weights(weights_dict)
        
        # Charger les paramètres de normalisation
        if 'x_mean' in data:
            self.x_mean = data['x_mean']
            self.x_std = data['x_std']
            self.y_mean = data['y_mean']
            self.y_std = data['y_std']
        
        # Charger le statut
        if 'is_trained' in data:
            self.is_trained = bool(data['is_trained'][0])
        
        print(f"✓ Modèle chargé depuis '{filepath}'")
    
    def __repr__(self):
        status = "entraîné" if self.is_trained else "non entraîné"
        return f"LinearAI(architecture={[self.input_size] + self.hidden_layers + [self.output_size]}, status='{status}')"
