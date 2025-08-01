"""
Module principal pour la génération de noms de domaine avec LLM.

Ce module utilise Ollama et Langchain pour générer des suggestions
de noms de domaine basées sur des descriptions business.
"""

import random


class CreationDataset:
    """
    Classe principale pour la création de dataset de noms de domaine.
    
    Cette classe gère la génération de suggestions de noms de domaine
    à partir de descriptions business en utilisant des modèles LLM.
    """

    def __init__(self):
        """Initialise le créateur de dataset avec des templates prédéfinis."""

        self.types_business = {
            "restaurant": {
                "descripteurs": ["bistro", "café", "restaurant", "brasserie", "traiteur"],
                "specialites": ["italien", "français", "africain", "asiatique", "végétarien", "bio", "gastronomique"],
                "lieux": ["centre-ville", "quartier résidentiel", "zone commerciale", "bord de mer"]
            },
            "technologie": {
                "descripteurs": ["startup", "entreprise tech", "développement", "consultance", "service"],
                "specialites": ["IA", "web", "mobile", "cloud", "cybersécurité", "blockchain"],
                "lieux": ["Silicon Valley", "Paris", "Londres", "télétravail", "incubateur"]
            },
            "commerce": {
                "descripteurs": ["boutique", "magasin", "commerce", "vente", "distribution"],
                "specialites": ["vêtements", "électronique", "livres", "sport", "beauté", "maison"],
                "lieux": ["en ligne", "centre commercial", "rue commerçante", "marché local"]
            },
            "services": {
                "descripteurs": ["cabinet", "agence", "consultant", "service", "assistance"],
                "specialites": ["comptabilité", "juridique", "marketing", "RH", "coaching", "traduction"],
                "lieux": ["bureau", "domicile", "chez le client", "en ligne"]
            },
            "sante": {
                "descripteurs": ["clinique", "cabinet", "centre", "pratique", "thérapie"],
                "specialites": ["médecine générale", "dentaire", "kinésithérapie", "psychologie", "nutrition"],
                "lieux": ["centre médical", "clinique privée", "hôpital", "domicile"]
            }
        }
        self.modificateurs = [
            "innovant", "moderne", "traditionnel", "premium", "abordable",
            "écologique", "durable", "artisanal", "personnalisé", "professionnel"
        ]
        self.extensions = [
            ".com", ".fr", ".org", ".net", ".eu", ".co", '.io', '.biz', '.info',
            '.tech', '.store', '.online', '.app', '.shop', '.site', '.cloud',
        ]
    
    def generer_description(self):
        """
        Génère une description business réaliste.
        
        Returns:
            str: Description business générée
        """
        
        # Sélectionner un type de business aléatoirement
        type_business = random.choice(list(self.types_business.keys()))
        business_info = self.types_business[type_business]
        
        # Construire la description
        descripteur = random.choice(business_info["descripteurs"])
        specialite = random.choice(business_info["specialites"])
        lieu = random.choice(business_info["lieux"])
        modificateur = random.choice(self.modificateurs) if random.random() > 0.5 else "" # ajouter de la complexité
        
        # Templates de descriptions
        templates = [
            f"{descripteur} {modificateur} spécialisé en {specialite} situé {lieu}".strip(),
            f"{modificateur} {descripteur} {specialite} dans {lieu}".strip(),
            f"Nouveau {descripteur} {specialite} {modificateur} à {lieu}".strip(),
            f"Service de {specialite} via {descripteur} {modificateur}".strip()
        ]
        
        description = random.choice(templates)
        description = ' '.join(description.split())  # Supprimer espaces multiples
        return description
