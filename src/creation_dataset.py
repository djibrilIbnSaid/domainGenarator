"""
Module principal pour la génération de noms de domaine avec LLM.

Ce module utilise Ollama et Langchain pour générer des suggestions
de noms de domaine basées sur des descriptions business.
"""

import random
from typing import Dict, List
from src.utils import  remove_accents, logger, generate_timestamp, sauvegarder_json


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
                "lieux": ["Silicon Valley", "Paris", "Londres", "Conakry", "télétravail", "incubateur"]
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
            ".com", ".fr", ".org", ".net", ".eu", ".co", ".io", ".ai", ".info",
            ".tech", ".store", ".online", ".app", ".shop", ".site", ".cloud", ".it",
        ]


    def generer_description(self) -> str:
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
    
    
    def generer_nom_domaine(self, description: str) -> str:
        """
        Génère un nom de domaine à partir d'une description business.
        
        Args:
            description (str): Description business
        
        Returns:
            str: Nom de domaine généré
        """
        
        # Extraire des mots clés de la description
        mots_cles = description.lower().split()
        mots_cles = [mot for mot in mots_cles if len(mot) > 3 and mot.isalpha() and mot not in
                     ['dans', 'avec', 'pour', 'situé', 'spécialisé', 'service']]
        
        # Stratégies de génération
        strategies = [
            self._combiner_mots_cles,
            self._ajouter_suffixe_business,
            self._utiliser_abreviations,
            self._generer_creatif
        ]
        
        strategie = random.choice(strategies)
        nom_base = strategie(mots_cles[:3])  # Limiter à 3 mots-clés
        extension = random.choice(self.extensions)

        nom_base = remove_accents(nom_base).lower()  # Supprimer les accents et mettre en minuscule
        return f"{nom_base}{extension}"
    
    
    def _combiner_mots_cles(self, mots_cles: List[str]) -> str:
        """Combine simplement 2-3 mots-clés."""
        if len(mots_cles) >= 2:
            return ''.join(mots_cles[:2])
        return mots_cles[0] if mots_cles else "example"
    
    def _ajouter_suffixe_business(self, mots_cles: List[str]) -> str:
        """Ajoute un suffixe business courant."""
        suffixes = ["pro", "expert", "plus", "hub", "lab", "group", "solutions"]
        base = mots_cles[0] if mots_cles else "business"
        suffixe = random.choice(suffixes)
        return f"{base}{suffixe}"
    
    def _utiliser_abreviations(self, mots_cles: List[str]) -> str:
        """Utilise des abréviations des mots-clés."""
        if not mots_cles:
            return "abc"
        
        # Prendre les premières lettres
        abbreviation = ''.join([mot[:2] for mot in mots_cles[:3]])
        return abbreviation
    
    def _generer_creatif(self, mots_cles: List[str]) -> str:
        """Génère un nom plus créatif."""
        prefixes = ["my", "get", "the", "pro", "smart", "quick", "best"]
        suffixes = ["ly", "fy", "ize", "hub", "box", "spot"]
        
        if mots_cles:
            base = mots_cles[0]
            if random.random() > 0.5:
                return f"{random.choice(prefixes)}{base}"
            else:
                return f"{base}{random.choice(suffixes)}"
        
        return f"{random.choice(prefixes)}{random.choice(['business', 'service', 'solution'])}"
    
    
    def creer_dataset_entrainement(self, taille: int = 1000) -> List[Dict[str, str]]:
        """
        Crée un dataset d'entraînement avec des paires description-domaine.
        
        Args:
            taille: Nombre d'exemples à générer
            
        Returns:
            List[Dict]: Dataset généré
        """
        logger.info(f"Génération d'un dataset de {taille} exemples...")
        
        dataset = []
        
        for i in range(taille):
            description = self.generer_description()
            domaine_exemple = self.generer_nom_domaine(description)
            
            exemple = {
                "id": i + 1,
                "description_business": description,
                "nom_domaine_exemple": domaine_exemple,
                "timestamp": generate_timestamp()
            }
            
            dataset.append(exemple)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Généré {i + 1}/{taille} exemples")
        
        return dataset
    
    
    def creer_cas_limites(self) -> List[Dict[str, str]]:
        """
        Crée des cas limites pour tester la robustesse du modèle.
        
        Returns:
            List[Dict]: Cas limites
        """
        return [
            # Descriptions très courtes
            {"description": "café", "categorie": "description_courte"},
            {"description": "tech", "categorie": "description_courte"},
            
            # Descriptions très longues
            {
                "description": "Une entreprise de développement de logiciels spécialisée dans les solutions d'intelligence artificielle et d'apprentissage automatique pour les grandes entreprises du secteur financier avec une expertise particulière en algorithmes de détection de fraude et systèmes de recommandation personnalisés utilisant des technologies de pointe comme TensorFlow et PyTorch",
                "categorie": "description_longue"
            },
            
            # Descriptions vagues
            {"description": "quelque chose de bien", "categorie": "description_vague"},
            {"description": "entreprise moderne", "categorie": "description_vague"},
            
            # Descriptions avec caractères spéciaux
            {"description": "café & restaurant français", "categorie": "caracteres_speciaux"},
            {"description": "entreprise high-tech à 100%", "categorie": "caracteres_speciaux"},
            
            # Descriptions en plusieurs langues
            {"description": "restaurant italiano molto buono", "categorie": "langue_etrangere"},
            {"description": "English consulting firm", "categorie": "langue_etrangere"},
            {"description": "restaurant Bilima", "categorie": "langue_etrangere"},
            
            # Concepts abstraits
            {"description": "plateforme de bonheur digital", "categorie": "concept_abstrait"},
            {"description": "solutions de bien-être énergétique", "categorie": "concept_abstrait"},
            
            # Niches très spécifiques
            {"description": "réparation de violons anciens", "categorie": "niche_specifique"},
            {"description": "élevage de papillons exotiques", "categorie": "niche_specifique"}
        ]
    
    
    def sauvegarder_dataset(self, nom_fichier: str = None) -> str:
        """
        Sauvegarde un dataset complet avec cas normaux et limites.
        
        Args:
            nom_fichier: Nom du fichier (optionnel)
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        if nom_fichier is None:
            nom_fichier = f"dataset_complet_{generate_timestamp()}.json"
        
        # Créer le dataset principal
        dataset_entrainement = self.creer_dataset_entrainement(1000)
        cas_limites = self.creer_cas_limites()
        
        dataset_complet = {
            "metadata": {
                "version": "1.0",
                "date_creation": generate_timestamp(),
                "nombre_exemples_entrainement": len(dataset_entrainement),
                "nombre_cas_limites": len(cas_limites)
            },
            "entrainement": dataset_entrainement,
            "cas_limites": cas_limites
        }
        
        chemin_fichier = f"../data/{nom_fichier}"
        sauvegarder_json(dataset_complet, chemin_fichier)
        
        logger.info(f"Dataset complet sauvegardé: {chemin_fichier}")
        return chemin_fichier


if __name__ == "__main__":
    createur = CreationDataset()
    chemin_dataset = createur.sauvegarder_dataset()
    print(f"Dataset créé: {chemin_dataset}")