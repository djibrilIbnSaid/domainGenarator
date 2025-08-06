"""
API REST FastAPI pour le générateur de noms de domaine.

Cette API expose les fonctionnalités du système de génération
et d'évaluation de noms de domaine via des endpoints REST.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from schemas import *
from src.generateur_nom_domaine import GenerateurNomsDomaine
from src.evaluateur_llm import EvaluateurLLM
from src.garde_fous import GardeFousSecurite
from src.utils import load_config

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
api_logger = logging.getLogger("api")

# Variables globales pour les services
generateur = None
evaluateur = None
garde_fous = None
config = None
statistiques_api = {
    "total_requests": 0,
    "successful_generations": 0,
    "blocked_requests": 0,
    "error_requests": 0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application."""
    # Démarrage
    global generateur, evaluateur, garde_fous, config
    
    api_logger.info("🚀 Démarrage du générateur de noms de domaine API...")
    
    try:
        # Charger la configuration
        config = load_config()
        api_logger.info("Configuration chargée")
        
        # Initialiser les services
        generateur = GenerateurNomsDomaine(garde_fous=True)
        api_logger.info("Générateur initialisé")
        
        evaluateur = EvaluateurLLM(type_evaluateur="ollama")
        api_logger.info("Évaluateur initialisé")
        
        garde_fous = GardeFousSecurite()
        api_logger.info("Garde-fous initialisés")
        
        api_logger.info("API prête à recevoir des requêtes")
        
    except Exception as e:
        api_logger.error(f"Erreur lors de l'initialisation: {e}")
        raise
    
    yield
    
    # Arrêt
    api_logger.info("Arrêt de l'API")


# Créer l'application FastAPI
app = FastAPI(
    title="Générateur de Noms de Domaine LLM",
    description="""
    API REST pour la génération et l'évaluation automatisées de noms de domaine 
    utilisant des modèles de langage (LLM).
    
    ## Fonctionnalités
    
    - **Génération**: Création de suggestions de noms de domaine basées sur une description business
    - **Évaluation**: Analyse automatisée de la qualité des noms de domaine
    - **Sécurité**: Filtrage de contenu inapproprié avec garde-fous intégrés
    - **Comparaison**: Comparaison de différentes versions de suggestions
    
    ## Modèles Utilisés
    
    - **Générateur**: Llama3.1 via Ollama
    - **Évaluateur**: LLM-as-a-Judge pour l'évaluation automatique
    - **Sécurité**: Système de filtrage personnalisé
    """,
    version="1.0.0",
    contact={
        "name": "AI Engineer",
        "email": "ai-engineer@example.com"
    },
    license_info={
        "name": "MIT License"
    },
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware pour les statistiques
@app.middleware("http")
async def statistiques_middleware(request: Request, call_next):
    """Middleware pour collecter les statistiques d'utilisation."""
    global statistiques_api
    
    start_time = time.time()
    statistiques_api["total_requests"] += 1
    
    response = await call_next(request)
    
    # Enregistrer les métriques
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def gestionnaire_erreur_global(request: Request, exc: Exception):
    """Gestionnaire d'erreurs global pour l'API."""
    global statistiques_api
    statistiques_api["error_requests"] += 1
    
    api_logger.error(f"Erreur non gérée: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Erreur interne du serveur",
            "code_erreur": "INTERNAL_ERROR",
            "timestamp": time.time()
        }
    )


# ENDPOINTS PRINCIPAUX

@app.get("/", response_model=Dict[str, Any])
async def racine():
    """
    Endpoint racine avec informations sur l'API.
    """
    return {
        "service": "Générateur de Noms de Domaine LLM",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": time.time(),
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=ReponseStatut)
async def verification_sante():
    """
    Vérification de l'état de santé des services.
    """
    global generateur, evaluateur, garde_fous, statistiques_api
    
    services_status = {}
    status_global = "healthy"
    
    # Vérifier le générateur
    try:
        if generateur is not None:
            info_modele = generateur.obtenir_info_modele()
            services_status["generateur"] = "operational"
        else:
            services_status["generateur"] = "unavailable"
            status_global = "degraded"
    except Exception:
        services_status["generateur"] = "error"
        status_global = "degraded"
    
    # Vérifier l'évaluateur
    try:
        if evaluateur is not None:
            info_evaluateur = evaluateur.obtenir_info_evaluateur()
            services_status["evaluateur"] = "operational"
        else:
            services_status["evaluateur"] = "unavailable"
            status_global = "degraded"
    except Exception:
        services_status["evaluateur"] = "error"
        status_global = "degraded"
    
    # Vérifier les garde-fous
    try:
        if garde_fous is not None:
            services_status["garde_fous"] = "operational"
        else:
            services_status["garde_fous"] = "unavailable"
    except Exception:
        services_status["garde_fous"] = "error"
    
    return ReponseStatut(
        status=status_global,
        version="1.0.0",
        timestamp=time.time(),
        services=services_status,
        statistiques={
            "total_requests": statistiques_api["total_requests"],
            "successful_generations": statistiques_api["successful_generations"],
            "blocked_requests": statistiques_api["blocked_requests"],
            "error_requests": statistiques_api["error_requests"],
            "uptime_seconds": time.time() - statistiques_api["start_time"]
        }
    )


@app.post("/generate", response_model=ReponseGeneration)
async def generer_noms_domaine(requete: RequeteGeneration):
    """
    Génère des suggestions de noms de domaine basées sur une description business.
    
    - **business_description**: Description de l'activité (5-500 caractères)
    - **nombre_suggestions**: Nombre de suggestions souhaitées (1-10)
    - **style**: Style de génération (professionnel, créatif, moderne, classique, court)
    - **avec_filtrage**: Activer les garde-fous de sécurité
    
    Retourne une liste de suggestions avec scores de confiance.
    """
    global generateur, statistiques_api
    
    if generateur is None:
        statistiques_api["error_requests"] += 1
        raise HTTPException(
            status_code=503,
            detail="Service de génération indisponible"
        )
    
    try:
        api_logger.info(f"Génération demandée: {requete.business_description[:50]}...")
        
        # Générer les suggestions
        resultats = generateur.generer_suggestions(
            description_business=requete.business_description,
            nombre_suggestions=requete.nombre_suggestions,
            style=requete.style.value,
            avec_filtrage=requete.avec_filtrage
        )
        
        # Traiter la réponse selon le statut
        if resultats["status"] == "success":
            statistiques_api["successful_generations"] += 1
            
            # Convertir en format de réponse
            suggestions = [
                SuggestionDomaine(
                    domaine=sug["domaine"],
                    confidence=sug["confidence"],
                    explication=sug.get("explication", "")
                )
                for sug in resultats["suggestions"]
            ]
            
            metadata = MetadataGeneration(**resultats["metadata"])
            
            return ReponseGeneration(
                suggestions=suggestions,
                status="success",
                message="Génération réussie",
                metadata=metadata,
                rapport_securite=resultats.get("rapport_securite")
            )
            
        elif resultats["status"] == "blocked":
            statistiques_api["blocked_requests"] += 1
            
            return ReponseGeneration(
                suggestions=[],
                status="blocked",
                message=resultats.get("message", "Contenu bloqué par les garde-fous"),
                rapport_securite=resultats.get("details_filtrage")
            )
            
        else:
            statistiques_api["error_requests"] += 1
            
            return ReponseGeneration(
                suggestions=[],
                status="error",
                message=resultats.get("message", "Erreur lors de la génération")
            )
            
    except Exception as e:
        statistiques_api["error_requests"] += 1
        api_logger.error(f"Erreur lors de la génération: {e}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération: {str(e)}"
        )


@app.post("/evaluate", response_model=ReponseEvaluation)
async def evaluer_noms_domaine(requete: RequeteEvaluation):
    """
    Évalue la qualité d'une liste de noms de domaine.
    
    - **business_description**: Description de l'activité business
    - **domaines**: Liste des noms de domaine à évaluer (1-10)
    - **avec_details**: Inclure les détails et recommandations
    
    Retourne des scores détaillés pour chaque critère d'évaluation.
    """
    global evaluateur, statistiques_api
    
    if evaluateur is None:
        raise HTTPException(
            status_code=503,
            detail="Service d'évaluation indisponible"
        )
    
    try:
        api_logger.info(f"Évaluation demandée: {len(requete.domaines)} domaines")
        
        # Évaluer les domaines
        resultats = evaluateur.evaluer_suggestions(
            description_business=requete.business_description,
            suggestions_domaines=requete.domaines,
            avec_details=requete.avec_details
        )
        
        if resultats["status"] == "success":
            # Convertir en format de réponse
            evaluations = [
                EvaluationDomaine(**eval_data)
                for eval_data in resultats["evaluations"]
            ]
            
            statistiques = StatistiquesEvaluation(**resultats["statistiques"])
            
            return ReponseEvaluation(
                status="success",
                evaluations=evaluations,
                statistiques=statistiques,
                recommandations=resultats.get("recommandations"),
                metadata=resultats.get("metadata")
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=resultats.get("message", "Erreur lors de l'évaluation")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Erreur lors de l'évaluation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'évaluation: {str(e)}"
        )


@app.get("/criteres", response_model=Dict[str, str])
async def obtenir_criteres_evaluation():
    """
    Retourne la liste des critères d'évaluation utilisés.
    """
    global evaluateur
    
    if evaluateur is None:
        raise HTTPException(
            status_code=503,
            detail="Service d'évaluation indisponible"
        )
    
    try:
        return evaluateur.obtenir_criteres_evaluation()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des critères: {str(e)}"
        )


@app.post("/batch/generate")
async def generer_batch(requetes: List[RequeteGeneration]):
    """
    Génère des suggestions pour plusieurs descriptions en lot.
    
    - **requetes**: Liste de requêtes de génération (max 10)
    
    Retourne les résultats pour chaque requête.
    """
    global generateur
    
    if generateur is None:
        raise HTTPException(
            status_code=503,
            detail="Service de génération indisponible"
        )
    
    if len(requetes) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 requêtes en lot"
        )
    
    try:
        api_logger.info(f"Génération en lot: {len(requetes)} requêtes")
        
        descriptions = [req.business_description for req in requetes]
        
        # Utiliser les paramètres de la première requête pour toutes
        params = {
            "nombre_suggestions": requetes[0].nombre_suggestions,
            "style": requetes[0].style.value,
            "avec_filtrage": requetes[0].avec_filtrage
        }
        
        resultats = generateur.generer_batch(descriptions, **params)
        
        return {
            "status": "success",
            "nombre_requetes": len(requetes),
            "resultats": resultats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        api_logger.error(f"Erreur lors de la génération en lot: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération en lot: {str(e)}"
        )


# Configuration personnalisée de la documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Générateur de Noms de Domaine LLM",
        version="1.0.0",
        description="""
        ## API Intelligence Artificielle pour Noms de Domaine
        
        Cette API utilise des modèles de langage avancés pour générer et évaluer 
        automatiquement des suggestions de noms de domaine pertinents.
        """,
        routes=app.routes,
    )
    
    # Ajouter des exemples personnalisés
    openapi_schema["info"]["x-logo"] = {
        "url": "https://via.placeholder.com/200x100?text=Domain+AI"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Point d'entrée pour le développement
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Démarrage du serveur de développement...")
    print("📚 Documentation disponible sur: http://localhost:8000/docs")
    print("🔍 Alternative ReDoc: http://localhost:8000/redoc")
    print("💡 Santé de l'API: http://localhost:8000/health")
    
    # Configuration du serveur
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "True").lower() == "true",
        log_level="info"
    )