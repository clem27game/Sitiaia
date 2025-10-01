"""
Module pour les IA génératives légères
"""

import numpy as np
from typing import List, Optional
import random


class GenerativeAI:
    """
    IA générative légère pour créer des noms et autres contenus
    
    Args:
        mode: Mode de génération ('name_generator', 'text_generator')
        **kwargs: Paramètres additionnels
        
    Example:
        >>> from sitiai import create
        >>> ai = create.ai('generative', mode='name_generator')
        >>> ai.load_data(['Alice', 'Bob', 'Charlie', 'David'])
        >>> ai.train(epochs=100)
        >>> new_name = ai.generate()
    """
    
    def __init__(self, mode: str = 'name_generator', **kwargs):
        self.mode = mode
        self.data = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.transition_matrix = None
        self.is_trained = False
        
    def load_data(self, data: List[str]):
        """
        Charge les données d'entraînement
        
        Args:
            data: Liste de chaînes de caractères pour l'entraînement
        """
        self.data = [d.lower() for d in data]
        
        # Créer le vocabulaire avec tokens spéciaux
        all_chars = set(''.join(self.data))
        all_chars.add('^')  # Token START
        all_chars.add('$')  # Token END
        
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
    def train(self, epochs: int = 100, ngram_size: int = 2):
        """
        Entraîne le modèle génératif
        
        Args:
            epochs: Nombre d'époques (non utilisé pour les n-grammes, mais conservé pour l'API)
            ngram_size: Taille des n-grammes (par défaut 2 = bigrammes)
        """
        if not self.data:
            raise ValueError("Aucune donnée chargée. Utilisez load_data() d'abord.")
        
        vocab_size = len(self.char_to_idx)
        self.transition_matrix = np.zeros((vocab_size, vocab_size))
        
        # Compter les transitions
        for word in self.data:
            # Ajouter tokens START (^) et END ($)
            extended_word = '^' + word + '$'
            
            for i in range(len(extended_word) - 1):
                curr_char = extended_word[i]
                next_char = extended_word[i + 1]
                
                curr_idx = self.char_to_idx[curr_char]
                next_idx = self.char_to_idx[next_char]
                
                self.transition_matrix[curr_idx, next_idx] += 1
        
        # Normaliser pour obtenir des probabilités
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Éviter division par zéro
        self.transition_matrix = self.transition_matrix / row_sums
        
        self.is_trained = True
        
    def generate(self, max_length: int = 20, temperature: float = 1.0) -> str:
        """
        Génère un nouveau nom ou texte
        
        Args:
            max_length: Longueur maximale de la génération
            temperature: Contrôle la créativité (plus élevé = plus créatif)
            
        Returns:
            Texte généré
        """
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné. Utilisez train() d'abord.")
        
        # Commencer par START (^)
        current_idx = self.char_to_idx['^']
        result = []
        
        for _ in range(max_length):
            # Obtenir les probabilités pour le prochain caractère
            probs = self.transition_matrix[current_idx].copy()
            
            # Appliquer la température
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs = probs / probs.sum()
            
            # Échantillonner le prochain caractère
            if probs.sum() == 0:
                break
                
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = self.idx_to_char[next_idx]
            
            # Si on atteint END ($), on s'arrête
            if next_char == '$':
                break
                
            result.append(next_char)
            current_idx = next_idx
        
        return ''.join(result).capitalize()
    
    def generate_batch(self, n: int = 5, max_length: int = 20, temperature: float = 1.0) -> List[str]:
        """
        Génère plusieurs résultats
        
        Args:
            n: Nombre de générations
            max_length: Longueur maximale de chaque génération
            temperature: Contrôle la créativité
            
        Returns:
            Liste de textes générés
        """
        return [self.generate(max_length, temperature) for _ in range(n)]
    
    def __repr__(self):
        status = "entraîné" if self.is_trained else "non entraîné"
        return f"GenerativeAI(mode='{self.mode}', status='{status}')"
