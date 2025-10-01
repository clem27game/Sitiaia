"""
Module central de Sitiai contenant l'API principale
"""

from .generative import GenerativeAI
from .linear import LinearAI


class CreateAPI:
    """API pour créer différents types d'IA"""
    
    def ai(self, ai_type: str, **kwargs):
        """
        Crée une nouvelle IA selon le type spécifié
        
        Args:
            ai_type: Type d'IA à créer ('generative' ou 'linear')
            **kwargs: Paramètres additionnels selon le type d'IA
            
        Returns:
            Instance d'IA correspondante
            
        Example:
            >>> import sitiai
            >>> ai = sitiai.create.ai('generative', mode='name_generator')
            >>> ai = sitiai.create.ai('linear', input_size=10, output_size=1)
        """
        if ai_type == 'generative':
            return GenerativeAI(**kwargs)
        elif ai_type == 'linear':
            return LinearAI(**kwargs)
        else:
            raise ValueError(f"Type d'IA non reconnu: {ai_type}. Utilisez 'generative' ou 'linear'")


# Instance globale de l'API create
create = CreateAPI()
