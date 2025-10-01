"""
Module contenant la couche de neurones SitiNEUR
Syntaxe inspirée de PyTorch mais simplifiée
"""

import numpy as np


class SitiNEUR:
    """
    Couche de neurones simplifiée pour Sitiai
    
    Args:
        input_size: Nombre d'entrées
        output_size: Nombre de sorties
        activation: Fonction d'activation ('relu', 'sigmoid', 'tanh', 'linear')
        
    Example:
        >>> from sitiai import SitiNEUR
        >>> layer = SitiNEUR(input_size=10, output_size=5, activation='relu')
        >>> output = layer.forward(input_data)
    """
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialisation des poids et biais (Xavier initialization)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)
        
        # Pour la rétropropagation
        self.last_input = None
        self.last_output = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propagation avant
        
        Args:
            x: Données d'entrée (shape: [batch_size, input_size])
            
        Returns:
            Sortie de la couche (shape: [batch_size, output_size])
        """
        self.last_input = x
        
        # Calcul: y = x @ weights + bias
        z = np.dot(x, self.weights) + self.bias
        
        # Application de la fonction d'activation
        self.last_output = self._apply_activation(z)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """
        Rétropropagation du gradient
        
        Args:
            grad_output: Gradient de la sortie
            learning_rate: Taux d'apprentissage
            
        Returns:
            Gradient pour la couche précédente
        """
        # Gradient de la fonction d'activation
        grad_activation = self._activation_gradient(self.last_output)
        grad_z = grad_output * grad_activation
        
        # Gradients des poids et biais
        grad_weights = np.dot(self.last_input.T, grad_z)
        grad_bias = np.sum(grad_z, axis=0)
        
        # Mise à jour des paramètres
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        # Gradient pour la couche précédente
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
    
    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """Applique la fonction d'activation"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Activation inconnue: {self.activation}")
    
    def _activation_gradient(self, output: np.ndarray) -> np.ndarray:
        """Calcule le gradient de la fonction d'activation"""
        if self.activation == 'relu':
            return (output > 0).astype(float)
        elif self.activation == 'sigmoid':
            return output * (1 - output)
        elif self.activation == 'tanh':
            return 1 - output ** 2
        elif self.activation == 'linear':
            return np.ones_like(output)
        else:
            raise ValueError(f"Activation inconnue: {self.activation}")
    
    def __repr__(self):
        return f"SitiNEUR(input_size={self.input_size}, output_size={self.output_size}, activation='{self.activation}')"
