"""
Schémas Pydantic pour l'API REST du générateur de noms de domaine.

Ce module définit les modèles de données pour les requêtes et réponses
de l'API utilisant Pydantic pour la validation et la documentation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class StyleGeneration(str, Enum):
    """Styles de génération disponibles."""
    professionnel = "professionnel"
    creatif = "creatif"
    moderne = "moderne"
    classique = "classique"
    court = "court"


class RequeteGeneration(BaseModel):
    """
    Modèle pour les requêtes de génération de noms de domaine.
    """
    business_description: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Description de l'activité business",
        example="restaurant africain moderne avec terrasse en centre-ville"
    )
    
    nombre_suggestions: Optional[int] = Field(
        5,
        ge=1,
        le=10,
        description="Nombre de suggestions à générer (1-10)",
        example=5
    )
    
    style: Optional[StyleGeneration] = Field(
        StyleGeneration.professionnel,
        description="Style de génération souhaité",
        example="professionnel"
    )
    
    avec_filtrage: Optional[bool] = Field(
        True,
        description="Activer les garde-fous de sécurité",
        example=True
    )
    
    @field_validator('business_description')
    @classmethod
    def valider_description(cls, v):
        """Valide que la description est appropriée."""
        if not v or not v.strip():
            raise ValueError('La description ne peut pas être vide')
        
        # Vérifications basiques
        if len(v.strip()) < 5:
            raise ValueError('La description doit contenir au moins 5 caractères')
        
        return v.strip()


class SuggestionDomaine(BaseModel):
    """
    Modèle pour une suggestion de nom de domaine.
    """
    domaine: str = Field(
        ...,
        description="Nom de domaine suggéré",
        example="fricbistro.com"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de confiance (0.0 à 1.0)",
        example=0.85
    )
    
    explication: Optional[str] = Field(
        None,
        description="Explication de la suggestion",
        example="Combine 'gusto' (goût) et 'arama' (suffixe moderne)"
    )


class MetadataGeneration(BaseModel):
    """
    Métadonnées sur la génération.
    """
    description_originale: str = Field(..., description="Description originale")
    style_utilise: str = Field(..., description="Style utilisé")
    nombre_demande: int = Field(..., description="Nombre de suggestions demandées")
    nombre_genere: int = Field(..., description="Nombre de suggestions générées")
    duree_generation_sec: float = Field(..., description="Durée de génération en secondes")
    modele_utilise: str = Field(..., description="Modèle LLM utilisé")
    filtrage_active: bool = Field(..., description="Filtrage de sécurité activé")


class ReponseGeneration(BaseModel):
    """
    Modèle pour les réponses de génération de noms de domaine.
    """
    suggestions: List[SuggestionDomaine] = Field(
        ...,
        description="Liste des suggestions générées"
    )
    
    status: str = Field(
        ...,
        description="Statut de la génération (success, blocked, error)",
        example="success"
    )
    
    message: Optional[str] = Field(
        None,
        description="Message explicatif si nécessaire",
        example="Génération réussie"
    )
    
    metadata: Optional[MetadataGeneration] = Field(
        None,
        description="Métadonnées sur la génération"
    )
    
    rapport_securite: Optional[Dict[str, Any]] = Field(
        None,
        description="Rapport de sécurité si filtrage activé"
    )


class RequeteEvaluation(BaseModel):
    """
    Modèle pour les requêtes d'évaluation de noms de domaine.
    """
    business_description: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Description de l'activité business"
    )
    
    domaines: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Liste des noms de domaine à évaluer",
        example=["gustorama.com", "bellavita.fr", "pastafino.com"]
    )
    
    avec_details: Optional[bool] = Field(
        True,
        description="Inclure les détails d'évaluation"
    )
    
    @field_validator('domaines')
    @classmethod
    def valider_domaines(cls, v):
        """Valide la liste des domaines."""
        if not v:
            raise ValueError('Au moins un domaine doit être fourni')
        
        # Validation basique des domaines
        for domaine in v:
            if not domaine or not domaine.strip():
                raise ValueError('Les domaines ne peuvent pas être vides')
            
            if '.' not in domaine:
                raise ValueError(f'Domaine invalide (pas d\'extension): {domaine}')
        
        return [d.strip().lower() for d in v]


class EvaluationDomaine(BaseModel):
    """
    Modèle pour l'évaluation d'un domaine.
    """
    domaine: str = Field(..., description="Nom de domaine évalué")
    pertinence: float = Field(..., ge=0.0, le=10.0, description="Score de pertinence (/10)")
    creativite: float = Field(..., ge=0.0, le=10.0, description="Score de créativité (/10)")
    memorabilite: float = Field(..., ge=0.0, le=10.0, description="Score de mémorabilité (/10)")
    disponibilite_estimee: float = Field(..., ge=0.0, le=10.0, description="Disponibilité estimée (/10)")
    professionnalisme: float = Field(..., ge=0.0, le=10.0, description="Score de professionnalisme (/10)")
    score_total: float = Field(..., ge=0.0, le=10.0, description="Score total (/10)")
    commentaire: Optional[str] = Field(None, description="Commentaire détaillé")


class StatistiquesEvaluation(BaseModel):
    """
    Statistiques sur les évaluations.
    """
    score_moyen: float = Field(..., description="Score moyen")
    score_median: float = Field(..., description="Score médian")
    score_min: float = Field(..., description="Score minimum")
    score_max: float = Field(..., description="Score maximum")
    ecart_type: float = Field(..., description="Écart-type")
    pertinence_moyenne: float = Field(..., description="Pertinence moyenne")
    creativite_moyenne: float = Field(..., description="Créativité moyenne")
    memorabilite_moyenne: float = Field(..., description="Mémorabilité moyenne")
    disponibilite_moyenne: float = Field(..., description="Disponibilité moyenne")
    professionnalisme_moyen: float = Field(..., description="Professionnalisme moyen")
    nombre_evaluations: int = Field(..., description="Nombre d'évaluations")


class ReponseEvaluation(BaseModel):
    """
    Modèle pour les réponses d'évaluation.
    """
    status: str = Field(..., description="Statut de l'évaluation")
    evaluations: List[EvaluationDomaine] = Field(..., description="Évaluations détaillées")
    statistiques: StatistiquesEvaluation = Field(..., description="Statistiques globales")
    recommandations: Optional[List[str]] = Field(None, description="Recommandations")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Métadonnées")


class ReponseErreur(BaseModel):
    """
    Modèle pour les réponses d'erreur.
    """
    status: str = Field("error", description="Statut d'erreur")
    message: str = Field(..., description="Message d'erreur")
    code_erreur: Optional[str] = Field(None, description="Code d'erreur spécifique")
    details: Optional[Dict[str, Any]] = Field(None, description="Détails supplémentaires")


class ReponseBloquee(BaseModel):
    """
    Modèle pour les réponses bloquées par les garde-fous.
    """
    status: str = Field("blocked", description="Statut bloqué")
    message: str = Field(..., description="Raison du blocage")
    suggestions: List[SuggestionDomaine] = Field([], description="Liste vide")
    details_filtrage: Optional[Dict[str, Any]] = Field(None, description="Détails du filtrage")


class ReponseStatut(BaseModel):
    """
    Modèle pour la réponse de statut de l'API.
    """
    status: str = Field("healthy", description="Statut du service")
    version: str = Field(..., description="Version de l'API")
    timestamp: float = Field(..., description="Timestamp du statut")
    services: Dict[str, str] = Field(..., description="Statut des services")
    statistiques: Optional[Dict[str, Any]] = Field(None, description="Statistiques d'utilisation")


