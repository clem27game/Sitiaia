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
        # Normaliser les entrées
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # S'assurer que y est 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples = X.shape[0]
        batch_size = batch_size or n_samples
        
        self.loss_history = []
        
        for epoch in range(epochs):
            # Mélanger les données
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Entraînement par mini-batches
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                # Forward pass
                predictions = self.forward(batch_X)
                
                # Calculer la perte (MSE)
                loss = np.mean((predictions - batch_y) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                grad = 2 * (predictions - batch_y) / batch_y.shape[0]
                
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Époque {epoch + 1}/{epochs}, Perte: {avg_loss:.6f}")
        
        self.is_trained = True
        
        if verbose:
            print(f"\nEntraînement terminé! Perte finale: {self.loss_history[-1]:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données
        
        Args:
            X: Données d'entrée
            
        Returns:
            Prédictions
        """
        if not self.is_trained:
            print("Attention: Le modèle n'est pas entraîné.")
        
        X = np.array(X, dtype=np.float32)
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
    
    def __repr__(self):
        status = "entraîné" if self.is_trained else "non entraîné"
        return f"LinearAI(architecture={[self.input_size] + self.hidden_layers + [self.output_size]}, status='{status}')"
