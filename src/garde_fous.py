"""
Module de garde-fous et filtrage de contenu pour le générateur de noms de domaine.

Ce module implémente des mécanismes de sécurité pour éviter la génération
de contenu inapproprié ou nuisible.

Il dispose deux methodes, une classique (algorithme de filtrage) et une basée sur un LLM
pour détecter les contenus sensibles.
"""

import re
import json
from typing import List, Dict
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from src.utils import load_config, logger

class GardeFousSecurite:
    """
    Classe pour implémenter les garde-fous de sécurité du système.
    """
    
    def __init__(self, use_llm: bool = False):
        """Initialise les listes de termes et patterns interdits."""
        
        self.use_llm = use_llm
        self.config = load_config()
        self.model = self.config.get("ollama_model", "llama3.1")
        
        if self.use_llm:
            try:
                self.llm = OllamaLLM(
                    model=self.model,
                    base_url=self.config["ollama_base_url"],
                    temperature=0,
                    num_predict=256
                )
                self.prompt = PromptTemplate(
                    input_variables=["description"],
                    template="""Tu es un système de sécurité automatisé pour la détection de contenus sensibles dans des descriptions business françaises.
                        Analyse la description suivante et réponds uniquement par un JSON strict (pas d'explications ni de texte libre) :
                        {{
                        "est_acceptable": true|false,
                        "score_risque": float entre 0 et 1,
                        "categories_detectees": [liste de catégories: "contenu_adulte", "violence", "drogue", "haine", "illegal", "domaine_sensible", "autre"],
                        "termes_problematiques": [liste de termes ou extraits détectés],
                        "suggestions_amelioration": [conseils ou suggestions pour améliorer la description]
                        }}
                        Description à analyser : "{description}"
                    """
                )
                self.chain = self.prompt | self.llm
                
                logger.info(f"Modèle {self.model} initialisé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du modèle: {e}")
                self.use_llm = False
        
        # Termes explicitement interdits (exemples génériques)
        self.termes_interdits = {
            'contenu_adulte': [
                'adult', 'porn', 'sex', 'nude', 'explicit',
                'adulte', 'sexe', 'nu', 'explicite', 'x'
            ],
            'violence': [
                'violence', 'weapon', 'kill', 'murder', 'terror',
                'arme', 'tuer', 'meurtre', 'terrorisme'
            ],
            'drogue': [
                'drug', 'cocaine', 'heroin', 'cannabis',
                'drogue', 'cocaïne', 'héroïne'
            ],
            'haine': [
                'hate', 'racist', 'nazi', 'supremacist',
                'haine', 'raciste', 'suprémaciste'
            ],
            'illegal': [
                'piracy', 'hack', 'fraud', 'scam',
                'piratage', 'fraude', 'arnaque'
            ]
        }

        self.patterns_suspects = [
            r'\b(free|gratuit)\s+(porn|adulte)\b',
            r'\b(hack|pirate)\s+(site|website)\b',
            r'\b(illegal|illégal)\s+(download|téléchargement)\b'
        ]
        self.domaines_sensibles = [
            'finance', 'banque', 'crédit', 'prêt',
            'médical', 'santé', 'médicament', 'thérapie',
            'juridique', 'avocat', 'conseil',
            'éducation', 'formation', 'diplôme'
        ]
        self.extensions_suspectes = ['.tk', '.ml', '.ga', '.cf']
    
    
    def analyser_description(self, description: str) -> Dict[str, any]:
        """
        Analyse une description business pour détecter du contenu inapproprié.
        
        Args:
            description: Description à analyser
            
        Returns:
            dict: Résultat de l'analyse avec score de risque
        """
        
        if self.use_llm:
            response = self.chain.invoke({"description": description})
            try:
                return self._parser_reponse_llm(response)
            except Exception as e:
                logger.error(f"Erreur analyse LLM, fallback vers méthode classique : {e}")

        # Méthode classique de filtrage
        description_lower = description.lower()
        
        # Initialiser le résultat
        resultat = {
            'est_acceptable': True,
            'score_risque': 0.0,
            'categories_detectees': [],
            'termes_problematiques': [],
            'suggestions_amelioration': []
        }
        
        # Vérifier les termes interdits
        for categorie, termes in self.termes_interdits.items():
            termes_trouves = [terme for terme in termes if terme in description_lower]
            if termes_trouves:
                resultat['categories_detectees'].append(categorie)
                resultat['termes_problematiques'].extend(termes_trouves)
                resultat['score_risque'] += 0.8  # Risque élevé
        
        # Vérifier les patterns suspects
        for pattern in self.patterns_suspects:
            if re.search(pattern, description_lower):
                resultat['score_risque'] += 0.6
                resultat['categories_detectees'].append('pattern_suspect')
        
        # Vérifier les domaines sensibles
        domaines_trouves = [domaine for domaine in self.domaines_sensibles 
                           if domaine in description_lower]
        if domaines_trouves:
            resultat['score_risque'] += 0.2  # Risque modéré
            resultat['categories_detectees'].append('domaine_sensible')
            resultat['suggestions_amelioration'].append(
                "Domaine sensible détecté. Validation supplémentaire recommandée."
            )
        
        # Déterminer l'acceptabilité
        if resultat['score_risque'] >= 0.7:
            resultat['est_acceptable'] = False
        elif resultat['score_risque'] >= 0.3:
            resultat['suggestions_amelioration'].append(
                "Description nécessitant une attention particulière."
            )
        
        # Normaliser le score
        resultat['score_risque'] = min(1.0, resultat['score_risque'])
        
        return resultat
    
    def filtrer_nom_domaine(self, nom_domaine: str) -> Dict[str, any]:
        """
        Filtre un nom de domaine pour détecter du contenu inapproprié.
        
        Args:
            nom_domaine: Nom de domaine à filtrer
            
        Returns:
            dict: Résultat du filtrage
        """
        nom_domaine_lower = nom_domaine.lower()
        
        resultat = {
            'est_acceptable': True,
            'raisons_rejet': [],
            'score_qualite': 1.0
        }
        
        # Vérifier les termes interdits dans le nom de domaine
        for categorie, termes in self.termes_interdits.items():
            for terme in termes:
                if terme in nom_domaine_lower:
                    resultat['est_acceptable'] = False
                    resultat['raisons_rejet'].append(f"Terme inapproprié: {terme}")
                    resultat['score_qualite'] = 0.0
        
        # Vérifier les extensions suspectes
        for extension in self.extensions_suspectes:
            if nom_domaine_lower.endswith(extension):
                resultat['score_qualite'] *= 0.5
                resultat['raisons_rejet'].append(f"Extension suspecte: {extension}")
        
        # Vérifier la longueur
        nom_sans_extension = nom_domaine_lower.split('.')[0]
        if len(nom_sans_extension) < 3:
            resultat['score_qualite'] *= 0.6
            resultat['raisons_rejet'].append("Nom de domaine trop court")
        elif len(nom_sans_extension) > 30:
            resultat['score_qualite'] *= 0.7
            resultat['raisons_rejet'].append("Nom de domaine trop long")
        
        # Vérifier les caractères
        if not re.match(r'^[a-z0-9.-]+$', nom_domaine_lower):
            resultat['est_acceptable'] = False
            resultat['raisons_rejet'].append("Caractères invalides détectés")
        
        return resultat
    
    def valider_liste_domaines(self, domaines: List[str]) -> List[Dict[str, any]]:
        """
        Valide une liste de noms de domaine.
        
        Args:
            domaines: Liste des noms de domaine à valider
            
        Returns:
            List[Dict]: Résultats de validation pour chaque domaine
        """
        resultats = []
        
        for domaine in domaines:
            validation = self.filtrer_nom_domaine(domaine)
            validation['domaine'] = domaine
            resultats.append(validation)
        
        return resultats
    
    def generer_rapport_securite(self, description: str, domaines: List[str]) -> Dict[str, any]:
        """
        Génère un rapport de sécurité complet.
        
        Args:
            description: Description business
            domaines: Liste de noms de domaine générés
            
        Returns:
            dict: Rapport de sécurité
        """
        # Analyser la description
        analyse_description = self.analyser_description(description)
        
        # Valider les domaines
        validations_domaines = self.valider_liste_domaines(domaines)
        
        # Calculer les statistiques
        domaines_acceptables = [v for v in validations_domaines if v['est_acceptable']]
        score_moyen_qualite = sum(v['score_qualite'] for v in validations_domaines) / len(validations_domaines) if validations_domaines else 0
        
        rapport = {
            'description_analysis': analyse_description,
            'domaines_valides': domaines_acceptables,
            'domaines_rejetes': [v for v in validations_domaines if not v['est_acceptable']],
            'statistiques': {
                'nombre_domaines_total': len(domaines),
                'nombre_domaines_acceptables': len(domaines_acceptables),
                'taux_acceptation': len(domaines_acceptables) / len(domaines) if domaines else 0,
                'score_qualite_moyen': score_moyen_qualite
            },
            'recommandations': self._generer_recommandations(analyse_description, validations_domaines)
        }
        
        return rapport
    
    def _generer_recommandations(self, analyse_description: Dict, validations_domaines: List[Dict]) -> List[str]:
        """
        Génère des recommandations basées sur l'analyse.
        
        Args:
            analyse_description: Résultat de l'analyse de description
            validations_domaines: Résultats de validation des domaines
            
        Returns:
            List[str]: Liste de recommandations
        """
        recommandations = []
        
        # Recommandations basées sur la description
        if not analyse_description['est_acceptable']:
            recommandations.append("Description contient du contenu inapproprié - révision nécessaire")
        elif analyse_description['score_risque'] > 0.3:
            recommandations.append("Description dans un domaine sensible - validation supplémentaire recommandée")
        
        # Recommandations basées sur les domaines
        domaines_rejetes = [v for v in validations_domaines if not v['est_acceptable']]
        if domaines_rejetes:
            recommandations.append(f"{len(domaines_rejetes)} domaine(s) rejeté(s) pour contenu inapproprié")
        
        score_moyen = sum(v['score_qualite'] for v in validations_domaines) / len(validations_domaines) if validations_domaines else 0
        if score_moyen < 0.7:
            recommandations.append("Qualité moyenne des domaines faible - amélioration suggérée")
        
        # Recommandations générales
        if len(validations_domaines) < 3:
            recommandations.append("Augmenter le nombre de suggestions pour plus de choix")
        
        return recommandations
    
    
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
            return json.loads(reponse_nettoyee)
                
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON: {e}")
            logger.error(f"Réponse brute: {reponse}")
            