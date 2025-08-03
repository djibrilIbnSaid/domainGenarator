"""
Module d'évaluation LLM-as-a-Judge pour les suggestions de noms de domaine.

Ce module implémente un système d'évaluation automatisé utilisant un LLM
pour juger la qualité des suggestions de noms de domaine.
"""

import time
import re
import json
from typing import Dict, List, Union

import numpy as np
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI, OpenAI

from src.utils import load_config, logger


class EvaluateurLLM:
    """
    Évaluateur automatique utilisant un LLM pour juger la qualité des noms de domaine.
    """
    
    def __init__(self, type_evaluateur: str = "ollama", model: str = None):
        """
        Initialise l'évaluateur LLM.
        
        Args:
            type_evaluateur: Type d'évaluateur ("ollama", "openai", "anthropic")
            modele: Nom du modèle à utiliser
        """
        self.config = load_config()
        self.type_evaluateur = type_evaluateur
        self.model = model or self.config.get("ollama_model_juge", "deepseek-r1")

        # Initialiser le LLM selon le type
        self._initialiser_llm()
        
        # Critères d'évaluation
        self.criteres_evaluation = {
            "pertinence": "Adéquation du nom de domaine à l'activité décrite",
            "creativite": "Originalité et mémorabilité du nom",
            "memorabilite": "Facilité de mémorisation et de pronunciation",
            "disponibilite_estimee": "Probabilité que le domaine soit disponible",
            "professionnalisme": "Aspect professionnel et sérieux du nom"
        }
        
        # Template de prompt pour l'évaluation
        self.template_evaluation = PromptTemplate(
            input_variables=["description_business", "domaines", "criteres"],
            template="""Tu es un expert en évaluation de noms de domaine avec 15 ans d'expérience.

                DESCRIPTION BUSINESS: {description_business}

                NOMS DE DOMAINE À ÉVALUER: {domaines}

                INSTRUCTIONS D'ÉVALUATION:
                Tu dois évaluer chaque nom de domaine selon ces critères (note sur 10):

                {criteres}

                Pour chaque domaine, considère:
                - La pertinence par rapport à l'activité
                - La facilité de mémorisation (court, sans caractères compliqués)
                - L'originalité sans être trop fantaisiste
                - La probabilité de disponibilité (éviter les noms trop génériques)
                - L'aspect professionnel et crédible

                BARÈME DE NOTATION:
                - 9-10: Excellent (parfaitement adapté, très mémorable)
                - 7-8: Très bon (bien adapté, quelques points d'amélioration)
                - 5-6: Correct (acceptable mais perfectible)
                - 3-4: Faible (problèmes significatifs)
                - 1-2: Très faible (inadapté ou problématique)

                FORMAT DE RÉPONSE (JSON strict):
                {{
                    "evaluations": [
                        {{
                            "domaine": "exemple.com",
                            "pertinence": 8.5,
                            "creativite": 7.0,
                            "memorabilite": 9.0,
                            "disponibilite_estimee": 6.5,
                            "professionnalisme": 8.0,
                            "score_total": 7.8,
                            "commentaire": "Explication concise des points forts et faibles"
                        }}
                    ]
                }}

                Réponds UNIQUEMENT avec le JSON, sans autre texte."""
            )
        
        # Créer la chaîne moderne avec RunnableSequence
        self.chain_evaluation = self.template_evaluation | self.llm
    
    
    def _initialiser_llm(self):
        """Initialise le LLM selon le type spécifié."""
        try:
            if self.type_evaluateur == "ollama":
                self.llm = OllamaLLM(
                    model=self.model,
                    base_url=self.config["ollama_base_url"],
                    temperature=0.1,
                    top_p=0.8,
                )
                logger.info(f"Évaluateur Ollama initialisé: {self.model}")
                
            elif self.type_evaluateur == "openai":
                self.llm = ChatOpenAI(
                    model=self.config.get("openai_model", "gpt-4o-mini"),
                    api_key=self.config.get("openai_api_key", ""),
                )
                logger.info(f"Évaluateur OpenAI initialisé: {self.model}")
                
            else:
                logger.warning(f"Type d'évaluateur inconnu: {self.type_evaluateur} | du coup par défaut on utilise Ollama")
                self.llm = OllamaLLM(
                    model=self.model,
                    base_url=self.config["ollama_base_url"],
                    temperature=0.1,
                    top_p=0.8,
                )
                self.type_evaluateur = "ollama"
                logger.info(f"Évaluateur Ollama initialisé: {self.model}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'évaluateur: {e}")
            raise
    
    
    def evaluer_suggestions(self, 
                          description_business: str, 
                          suggestions_domaines: List[str],
                          avec_details: bool = True) -> Dict[str, any]:
        """
        Évalue une liste de suggestions de noms de domaine.
        
        Args:
            description_business: Description de l'entreprise
            suggestions_domaines: Liste des noms de domaine à évaluer
            avec_details: Inclure les détails d'évaluation
            
        Returns:
            dict: Résultats d'évaluation avec scores et commentaires
        """
        
        if not suggestions_domaines:
            return {
                "status": "error",
                "message": "Aucune suggestion à évaluer",
                "evaluations": []
            }
        
        try:
            debut = time.time()
            
            # Préparer les données pour le prompt
            domaines_formates = "\n".join([f"- {domaine}" for domaine in suggestions_domaines])
            criteres_formates = "\n".join([f"- {nom}: {desc}" for nom, desc in self.criteres_evaluation.items()])
            
            # Exécuter l'évaluation avec la nouvelle approche
            reponse = self.chain_evaluation.invoke({
                "description_business": description_business,
                "domaines": domaines_formates,
                "criteres": criteres_formates
            })
            reponse = reponse if isinstance(reponse, str) else reponse.content

            duree_evaluation = time.time() - debut
            
            # Parser la réponse
            evaluations = self._parser_reponse_evaluation(reponse, suggestions_domaines)
            
            # Calculer les statistiques
            statistiques = self._calculer_statistiques_evaluation(evaluations)
            
            resultats = {
                "status": "success",
                "evaluations": evaluations,
                "statistiques": statistiques,
                "metadata": {
                    "description_originale": description_business,
                    "nombre_domaines_evalues": len(suggestions_domaines),
                    "duree_evaluation_sec": round(duree_evaluation, 2),
                    "evaluateur_utilise": f"{self.type_evaluateur}_{self.model}",
                    "timestamp": time.time()
                }
            }
            
            if avec_details:
                resultats["criteres_utilises"] = self.criteres_evaluation
                resultats["recommandations"] = self._generer_recommandations(evaluations)
            
            logger.info(f"Évaluation terminée: score moyen {statistiques['score_moyen']:.2f}/10")
            return resultats
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation: {e}")
            return {
                "status": "error",
                "message": f"Erreur lors de l'évaluation: {str(e)}",
                "evaluations": []
            }
    
    
    def _parser_reponse_evaluation(self, reponse: str, domaines_originaux: List[str], is_ollama: bool = True) -> List[Dict[str, any]]:
        """
        Parse la réponse JSON de l'évaluateur.
        
        Args:
            reponse: Réponse brute du LLM
            domaines_originaux: Liste des domaines originaux pour validation
            is_ollama: Indique si la réponse provient d'Ollama (pour ajuster le parsing)
            
        Returns:
            List[Dict]: Évaluations parsées et validées
        """
        try:
            # Nettoyer la réponse
            reponse_nettoyee = reponse.strip()
            reponse_nettoyee = re.sub(r'<think>.*?</think>', '', reponse_nettoyee, flags=re.DOTALL)
            reponse_nettoyee = reponse_nettoyee.strip()
            if reponse_nettoyee.startswith("```json"):
                reponse_nettoyee = reponse_nettoyee[7:]
            if reponse_nettoyee.endswith("```"):
                reponse_nettoyee = reponse_nettoyee[:-3]
            
            # Parser le JSON
            data = json.loads(reponse_nettoyee)
            
            if "evaluations" in data and isinstance(data["evaluations"], list):
                evaluations_validees = []
                
                for eval_data in data["evaluations"]:
                    # Valider la structure
                    if not isinstance(eval_data, dict) or "domaine" not in eval_data:
                        continue
                    
                    # Valider que le domaine est dans la liste originale
                    if eval_data["domaine"] not in domaines_originaux:
                        continue
                    
                    # Valider et normaliser les scores
                    evaluation_validee = {
                        "domaine": eval_data["domaine"],
                        "pertinence": self._normaliser_score(eval_data.get("pertinence", 5)),
                        "creativite": self._normaliser_score(eval_data.get("creativite", 5)),
                        "memorabilite": self._normaliser_score(eval_data.get("memorabilite", 5)),
                        "disponibilite_estimee": self._normaliser_score(eval_data.get("disponibilite_estimee", 5)),
                        "professionnalisme": self._normaliser_score(eval_data.get("professionnalisme", 5)),
                        "commentaire": eval_data.get("commentaire", "")
                    }
                    
                    # Calculer le score total
                    scores = [
                        evaluation_validee["pertinence"],
                        evaluation_validee["creativite"], 
                        evaluation_validee["memorabilite"],
                        evaluation_validee["disponibilite_estimee"],
                        evaluation_validee["professionnalisme"]
                    ]
                    evaluation_validee["score_total"] = round(np.mean(scores), 1)
                    
                    evaluations_validees.append(evaluation_validee)
                
                return evaluations_validees
            else:
                logger.warning("Format de réponse d'évaluation inattendu")
                return self._generer_evaluations_fallback(domaines_originaux)
                
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON de l'évaluation: {e}")
            return self._generer_evaluations_fallback(domaines_originaux)
        
        except Exception as e:
            logger.error(f"Erreur lors du parsing de l'évaluation: {e}")
            return self._generer_evaluations_fallback(domaines_originaux)
    
    def _normaliser_score(self, score: Union[int, float, str]) -> float:
        """
        Normalise un score pour qu'il soit entre 0 et 10.
        
        Args:
            score: Score à normaliser
            
        Returns:
            float: Score normalisé
        """
        try:
            score_float = float(score)
            return max(0.0, min(10.0, round(score_float, 1)))
        except (ValueError, TypeError):
            return 5.0
    
    def _generer_evaluations_fallback(self, domaines: List[str]) -> List[Dict[str, any]]:
        """
        Génère des évaluations de fallback en cas d'échec du parsing.
        
        Args:
            domaines: Liste des domaines à évaluer
            
        Returns:
            List[Dict]: Évaluations de fallback
        """
        evaluations_fallback = []
        
        for domaine in domaines:
            # Score de base basé sur des heuristiques simples
            nom_sans_extension = domaine.split(".")[0] if "." in domaine else domaine
            
            # Heuristiques basiques
            pertinence = 6.0
            creativite = 7.0 if len(nom_sans_extension) <= 12 else 5.0
            memorabilite = 8.0 if len(nom_sans_extension) <= 10 else 6.0
            disponibilite = 7.0 if "-" not in nom_sans_extension else 8.0
            professionnalisme = 7.0 if not any(char.isdigit() for char in nom_sans_extension) else 6.0
            
            scores = [pertinence, creativite, memorabilite, disponibilite, professionnalisme]
            score_total = round(np.mean(scores), 1)
            
            evaluation_fallback = {
                "domaine": domaine,
                "pertinence": pertinence,
                "creativite": creativite,
                "memorabilite": memorabilite,
                "disponibilite_estimee": disponibilite,
                "professionnalisme": professionnalisme,
                "score_total": score_total,
                "commentaire": "Évaluation automatique de fallback"
            }
            
            evaluations_fallback.append(evaluation_fallback)
        
        logger.warning(f"Utilisation d'évaluations de fallback pour {len(domaines)} domaines")
        return evaluations_fallback
    
    def _calculer_statistiques_evaluation(self, evaluations: List[Dict[str, any]]) -> Dict[str, float]:
        """
        Calcule des statistiques sur les évaluations.
        
        Args:
            evaluations: Liste des évaluations
            
        Returns:
            dict: Statistiques calculées
        """
        if not evaluations:
            return {}
        
        scores_totaux = [eval_data["score_total"] for eval_data in evaluations]
        scores_pertinence = [eval_data["pertinence"] for eval_data in evaluations]
        scores_creativite = [eval_data["creativite"] for eval_data in evaluations]
        scores_memorabilite = [eval_data["memorabilite"] for eval_data in evaluations]
        scores_disponibilite = [eval_data["disponibilite_estimee"] for eval_data in evaluations]
        scores_professionnalisme = [eval_data["professionnalisme"] for eval_data in evaluations]
        
        return {
            "score_moyen": round(np.mean(scores_totaux), 2),
            "score_median": round(np.median(scores_totaux), 2),
            "score_min": round(min(scores_totaux), 2),
            "score_max": round(max(scores_totaux), 2),
            "ecart_type": round(np.std(scores_totaux), 2),
            "pertinence_moyenne": round(np.mean(scores_pertinence), 2),
            "creativite_moyenne": round(np.mean(scores_creativite), 2),
            "memorabilite_moyenne": round(np.mean(scores_memorabilite), 2),
            "disponibilite_moyenne": round(np.mean(scores_disponibilite), 2),
            "professionnalisme_moyen": round(np.mean(scores_professionnalisme), 2),
            "nombre_evaluations": len(evaluations)
        }
    
    def _generer_recommandations(self, evaluations: List[Dict[str, any]]) -> List[str]:
        """
        Génère des recommandations basées sur les évaluations.
        
        Args:
            evaluations: Liste des évaluations
            
        Returns:
            List[str]: Liste de recommandations
        """
        if not evaluations:
            return ["Aucune évaluation disponible pour générer des recommandations"]
        
        recommandations = []
        
        # Analyser les scores moyens par critère
        stats = self._calculer_statistiques_evaluation(evaluations)
        
        # Recommandations basées sur les critères faibles
        if stats["pertinence_moyenne"] < 6.0:
            recommandations.append("Améliorer la pertinence: choisir des noms plus en lien avec l'activité")
        
        if stats["creativite_moyenne"] < 6.0:
            recommandations.append("Améliorer la créativité: explorer des noms plus originaux et mémorables")
        
        if stats["memorabilite_moyenne"] < 6.0:
            recommandations.append("Améliorer la mémorabilité: privilégier des noms plus courts et simples")

        if stats["disponibilite_moyenne"] < 6.0:
            recommandations.append("Améliorer la disponibilité: éviter les noms trop génériques")

        if stats["professionnalisme_moyen"] < 6.0:
            recommandations.append("Améliorer le professionnalisme: choisir des noms plus sérieux et crédibles")
        
        # Recommandations basées sur la variabilité
        if stats["ecart_type"] > 2.0:
            recommandations.append("Homogénéiser la qualité: les suggestions varient beaucoup en qualité")
        
        # Recommandations basées sur le score global
        if stats["score_moyen"] < 5.0:
            recommandations.append("⚡ Revoir la stratégie de génération: score global faible")
        elif stats["score_moyen"] >= 8.0:
            recommandations.append("Excellente qualité globale: maintenir cette approche")
        
        # Identifier le meilleur et le pire
        meilleur = max(evaluations, key=lambda x: x["score_total"])
        pire = min(evaluations, key=lambda x: x["score_total"])
        
        if meilleur["score_total"] >= 8.0:
            recommandations.append(f"Excellent choix: {meilleur['domaine']} ({meilleur['score_total']}/10)")

        if pire["score_total"] <= 4.0:
            recommandations.append(f"À éviter: {pire['domaine']} ({pire['score_total']}/10)")

        return recommandations if recommandations else ["Suggestions de bonne qualité globale"]
    
    
    def obtenir_info_evaluateur(self) -> Dict[str, any]:
        """
        Retourne des informations sur l'évaluateur.
        
        Returns:
            dict: Informations sur l'évaluateur
        """
        return {
            "type_evaluateur": self.type_evaluateur,
            "modele_utilise": self.model,
            "criteres_evaluation": list(self.criteres_evaluation.keys()),
            "nombre_criteres": len(self.criteres_evaluation),
            "config": self.config
        }
    
    
    def obtenir_criteres_evaluation(self) -> Dict[str, str]:
        """
        Retourne les critères d'évaluation utilisés.
        
        Returns:
            dict: Critères et leurs descriptions
        """
        return self.criteres_evaluation.copy()