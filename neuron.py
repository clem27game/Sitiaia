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
        
        # Initialisation des poids et biais (He initialization pour ReLU, Xavier pour autres)
        if activation == 'relu':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        self.bias = np.zeros(output_size)
        
        # Pour la rétropropagation
        self.last_input = None
        self.last_output = None
        self.last_z = None
        
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
        self.last_z = np.dot(x, self.weights) + self.bias
        
        # Application de la fonction d'activation
        self.last_output = self._apply_activation(self.last_z)
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
        # Clip gradient output to prevent explosion
        grad_output = np.clip(grad_output, -10, 10)
        
        # Gradient de la fonction d'activation
        if self.activation == 'relu':
            grad_activation = (self.last_z > 0).astype(float)
        else:
            grad_activation = self._activation_gradient(self.last_output)
        
        grad_z = grad_output * grad_activation
        
        # Clip intermediate gradients
        grad_z = np.clip(grad_z, -10, 10)
        
        # Gradients des poids et biais
        grad_weights = np.dot(self.last_input.T, grad_z)
        grad_bias = np.sum(grad_z, axis=0)
        
        # Clip parameter gradients
        grad_weights = np.clip(grad_weights, -1, 1)
        grad_bias = np.clip(grad_bias, -1, 1)
        
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
    
    def get_weights(self):
        """Retourne les poids et biais de la couche"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }
    
    def set_weights(self, weights_dict):
        """Définit les poids et biais de la couche"""
        self.weights = weights_dict['weights'].copy()
        self.bias = weights_dict['bias'].copy()
    
    def __repr__(self):
        return f"SitiNEUR(input_size={self.input_size}, output_size={self.output_size}, activation='{self.activation}')"
