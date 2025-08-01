"""
Utilitaires généraux pour le projet de génération de noms de domaine.

Ce module contient des fonctions utilitaires partagées entre
les différents composants du système.
"""

import os

def load_config():
    """
    Charge la configuration du projet depuis un fichier.

    Returns:
        dict: Configuration chargée
    """
    return {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1"),
        "debug": os.getenv("DEBUG", "True").lower() == "true",
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    }