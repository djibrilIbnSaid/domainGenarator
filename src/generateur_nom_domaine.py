"""
Module principal pour la génération de noms de domaine avec LLM.

Ce module utilise Ollama et Langchain pour générer des suggestions
de noms de domaine basées sur des descriptions business.
"""
import time
import json
from typing import Dict, List

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.garde_fous import GardeFousSecurite
from src.utils import load_config, logger, nettoyer_nom_domaine


class GenerateurNomsDomaine:
    def __init__(self, model: str = "llama3.1", garde_fous: bool = True, garde_fous_use_llm: bool = False):
        """
        Initialise le générateur avec le modèle spécifié.
        
        Args:
            modele: Nom du modèle Ollama à utiliser
            garde_fous: Activer les garde-fous de sécurité
        """
        self.config = load_config()
        self.model = model or self.config["ollama_model"]
        
        try:
            self.llm = OllamaLLM(
                model=self.model,
                base_url=self.config["ollama_base_url"],
                temperature=0.7,
                top_p=0.9,
                num_predict=256
            )
            logger.info(f"Modèle {self.model} initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle: {e}")
            raise
            
        self.garde_fous = GardeFousSecurite(garde_fous_use_llm) if garde_fous else None
        
        self.prompt = PromptTemplate(
            input_variables=["description", "nombre_suggestions", "style"],
            template="""Tu es un expert en création de noms de domaine pour entreprises.

                DESCRIPTION BUSINESS: {description}

                INSTRUCTIONS:
                - Génère exactement {nombre_suggestions} suggestions de noms de domaine
                - Style demandé: {style}
                - Les noms doivent être mémorables, faciles à épeler et à retenir
                - Évite les caractères spéciaux et les tirets multiples
                - Privilégie les extensions .com, .fr, .org, .net
                - Assure-toi que les noms sont appropriés et professionnels

                FORMAT DE RÉPONSE (JSON uniquement):
                {{
                    "suggestions": [
                        {{"domaine": "exemple1.com", "explication": "Description courte"}},
                        {{"domaine": "exemple2.fr", "explication": "Description courte"}},
                        ...
                    ]
                }}

                Réponds uniquement avec le JSON, sans autre texte."""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        # Styles de génération disponibles
        self.styles_disponibles = {
            "professionnel": "Noms sérieux et corporatifs",
            "creatif": "Noms originaux et mémorables", 
            "moderne": "Noms tendance avec suffixes tech",
            "classique": "Noms traditionnels et établis",
            "court": "Noms courts et percutants"
        }
    
    
    def generer_suggestions(self, 
                          description_business: str, 
                          nombre_suggestions: int = 5,
                          style: str = "professionnel",
                          avec_filtrage: bool = True) -> Dict[str, any]:
        """
        Génère des suggestions de noms de domaine.
        
        Args:
            description_business: Description de l'entreprise
            nombre_suggestions: Nombre de suggestions à générer
            style: Style de génération souhaité
            avec_filtrage: Appliquer les garde-fous de sécurité
            
        Returns:
            dict: Résultats avec suggestions et métadonnées
        """
        logger.info(f"Génération de {nombre_suggestions} | suggestions pour: {description_business[:50]}...")
        
        # Vérifier les garde-fous si activés
        if avec_filtrage and self.garde_fous:
            analyse_description = self.garde_fous.analyser_description(description_business)
            if not analyse_description['est_acceptable']:
                return {
                    "suggestions": [],
                    "status": "blocked",
                    "message": "Description contient du contenu inapproprié",
                    "details_filtrage": analyse_description
                }
        
        # Valider les paramètres
        if style not in self.styles_disponibles:
            style = "professionnel"
            
        nombre_suggestions = max(1, min(nombre_suggestions, 10))  # Limiter entre 1 et 10
        
        try:
            # Générer les suggestions avec la nouvelle syntaxe
            debut = time.time()
            
            reponse = self.chain.invoke({
                "description": description_business,
                "nombre_suggestions": nombre_suggestions,
                "style": self.styles_disponibles[style]
            })
            
            duree_generation = time.time() - debut
            
            # Parser la réponse JSON
            suggestions_brutes = self._parser_reponse_llm(reponse)
            
            # Nettoyer et valider les suggestions
            suggestions_validees = self._valider_suggestions(
                suggestions_brutes, avec_filtrage
            )
            
            # Calculer les scores de confiance
            suggestions_avec_scores = self._calculer_scores_confiance(
                suggestions_validees, description_business
            )
            
            resultats = {
                "suggestions": suggestions_avec_scores,
                "status": "success",
                "metadata": {
                    "description_originale": description_business,
                    "style_utilise": style,
                    "nombre_demande": nombre_suggestions,
                    "nombre_genere": len(suggestions_avec_scores),
                    "duree_generation_sec": round(duree_generation, 2),
                    "modele_utilise": self.model,
                    "filtrage_active": avec_filtrage
                }
            }
            
            # Ajouter le rapport de sécurité si filtrage activé
            if avec_filtrage and self.garde_fous:
                domaines = [s["domaine"] for s in suggestions_avec_scores]
                rapport_securite = self.garde_fous.generer_rapport_securite(
                    description_business, domaines
                )
                resultats["rapport_securite"] = rapport_securite
            
            logger.info(f"Génération terminée: {len(suggestions_avec_scores)} suggestions créées")
            return resultats
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            return {
                "suggestions": [],
                "status": "error",
                "message": f"Erreur lors de la génération: {str(e)}"
            }
    
    
    def _parser_reponse_llm(self, reponse: str) -> List[Dict[str, str]]:
        """
        Parse la réponse JSON du LLM.
        
        Args:
            reponse: Réponse brute du LLM
            
        Returns:
            List[Dict]: Suggestions parsées
        """
        try:
            # Nettoyer la réponse (supprimer les backticks, etc.)
            reponse_nettoyee = reponse.strip()
            if reponse_nettoyee.startswith("```json"):
                reponse_nettoyee = reponse_nettoyee[7:]
            if reponse_nettoyee.endswith("```"):
                reponse_nettoyee = reponse_nettoyee[:-3]
            
            # Parser le JSON
            data = json.loads(reponse_nettoyee)
            
            if "suggestions" in data and isinstance(data["suggestions"], list):
                return data["suggestions"]
            else:
                logger.warning("Format de réponse LLM inattendu")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON: {e}")
            logger.error(f"Réponse brute: {reponse}")
            
            # Fallback: essayer d'extraire les domaines avec regex
            return self._extraire_domaines_fallback(reponse)
        
        except Exception as e:
            logger.error(f"Erreur lors du parsing: {e}")
            return []
    
    def _extraire_domaines_fallback(self, reponse: str) -> List[Dict[str, str]]:
        """
        Méthode de fallback pour extraire des domaines de la réponse.
        
        Args:
            reponse: Réponse brute du LLM
            
        Returns:
            List[Dict]: Domaines extraits
        """
        import re
        
        # Pattern pour détecter des noms de domaine
        pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.(com|fr|org|net|eu|co|ai|com|fr|org|net|eu|co|io|ai|info|tech|store|online|app|shop|site|cloud|it)\b'
        domaines_trouves = re.findall(pattern, reponse.lower())
        
        suggestions = []
        for domaine_match in domaines_trouves:
            domaine = domaine_match[0] + "." + domaine_match[1]
            suggestions.append({
                "domaine": domaine,
                "explication": "Extrait automatiquement"
            })
        
        return suggestions[:5]
    
    
    def _calculer_scores_confiance(self, suggestions: List[Dict[str, str]], description: str) -> List[Dict[str, any]]:
        """
        Calcule des scores de confiance pour les suggestions.
        
        Args:
            suggestions: Suggestions validées
            description: Description business originale
            
        Returns:
            List[Dict]: Suggestions avec scores de confiance
        """
        mots_cles_description = set(description.lower().split())
        suggestions_avec_scores = []
        
        for suggestion in suggestions:
            domaine = suggestion["domaine"]
            nom_sans_extension = domaine.split(".")[0]
            
            # Facteurs de score
            score = 0.5  # Score de base
            
            # Longueur optimale (6-15 caractères)
            longueur = len(nom_sans_extension)
            if 6 <= longueur <= 15:
                score += 0.2
            elif longueur < 6:
                score += 0.1
            
            # Facilité de mémorisation (pas de tirets multiples, caractères répétitifs)
            if "--" not in domaine and not any(c * 3 in domaine for c in "abcdefghijklmnopqrstuvwxyz"):
                score += 0.1
            
            # Pertinence par rapport à la description
            mots_domaine = set([nom_sans_extension] + nom_sans_extension.split("-"))
            correspondances = len(mots_cles_description.intersection(mots_domaine))
            if correspondances > 0:
                score += min(0.2, correspondances * 0.1)
            
            # Extension préférée
            if domaine.endswith((".com", ".fr")):
                score += 0.1
            
            # Normaliser le score entre 0.1 et 1.0
            score = max(0.1, min(1.0, score))
            
            suggestion_avec_score = {
                "domaine": domaine,
                "confidence": round(score, 2),
                "explication": suggestion["explication"]
            }
            
            suggestions_avec_scores.append(suggestion_avec_score)
        
        # Trier par score de confiance décroissant
        suggestions_avec_scores.sort(key=lambda x: x["confidence"], reverse=True)
        
        return suggestions_avec_scores
    
    
    def _valider_suggestions(self, suggestions: List[Dict[str, str]], avec_filtrage: bool) -> List[Dict[str, str]]:
        """
        Valide et nettoie les suggestions.
        
        Args:
            suggestions: Suggestions brutes
            avec_filtrage: Appliquer le filtrage de sécurité
            
        Returns:
            List[Dict]: Suggestions validées
        """
        suggestions_validees = []
        
        for suggestion in suggestions:
            if not isinstance(suggestion, dict) or "domaine" not in suggestion:
                continue
            
            domaine = nettoyer_nom_domaine(suggestion["domaine"])
            
            # Vérifications de base
            if len(domaine) < 4 or len(domaine) > 50:
                continue
            
            if not "." in domaine:
                continue
            
            # Filtrage de sécurité si activé
            if avec_filtrage and self.garde_fous:
                validation = self.garde_fous.filtrer_nom_domaine(domaine)
                if not validation["est_acceptable"]:
                    continue
            
            suggestion_validee = {
                "domaine": domaine,
                "explication": suggestion.get("explication", "")
            }
            
            suggestions_validees.append(suggestion_validee)
        
        return suggestions_validees

    
    def obtenir_info_modele(self) -> Dict[str, any]:
        """
        Obtient des informations sur le modèle utilisé.
        
        Returns:
            dict: Informations sur le modèle
        """
        return {
            "nom_modele": self.model,
            "base_url": self.config["ollama_base_url"],
            "styles_disponibles": list(self.styles_disponibles.keys()),
            "garde_fous_actifs": self.garde_fous is not None,
            "config": self.config
        }