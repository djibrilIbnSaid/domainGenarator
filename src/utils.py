"""
Utilitaires généraux pour le projet de génération de noms de domaine.

Ce module contient des fonctions utilitaires partagées entre
les différents composants du système.
"""

import os
from typing import Any, Dict
import unicodedata
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def remove_accents(input_str: str) -> str:
    """
    Supprime les accents d'une chaîne de caractères.

    Args:
        input_str (str): Chaîne d'entrée

    Returns:
        str: Chaîne sans accents
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def generate_timestamp() -> str:
    """
    Génère un timestamp pour nommer les fichiers.
    
    Returns:
        str: Timestamp au format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def sauvegarder_json(data: Dict[Any, Any], filepath: str) -> None:
    """
    Sauvegarde des données au format JSON.
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin du fichier de destination
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Données sauvegardées dans {filepath}")
    

def charger_json(filepath: str) -> Dict[Any, Any]:
    """
    Charge des données depuis un fichier JSON.
    
    Args:
        filepath: Chemin du fichier à charger
        
    Returns:
        dict: Données chargées
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Données chargées depuis {filepath}")
        return data
    except FileNotFoundError:
        logger.warning(f"Fichier {filepath} non trouvé")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Erreur de décodage JSON dans {filepath}")
        return {}


def nettoyer_nom_domaine(domaine: str) -> str:
    """
    Nettoie et normalise un nom de domaine.
    
    Args:
        domaine: Nom de domaine brut
        
    Returns:
        str: Nom de domaine nettoyé
    """
    # Supprimer les espaces et caractères spéciaux
    domaine = domaine.strip().lower()
    
    # Supprimer http:// ou https://
    if domaine.startswith(('http://', 'https://')):
        domaine = domaine.split('://', 1)[1]
    
    # Supprimer www.
    if domaine.startswith('www.'):
        domaine = domaine[4:]
    
    # Garder seulement les caractères alphanumériques, tirets et points
    domaine_nettoye = ''.join(c for c in domaine if c.isalnum() or c in '.-')
    
    return domaine_nettoye