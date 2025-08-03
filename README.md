# Générateur de Noms de Domaine avec LLM
## Projet AI Engineer - Évaluation et Amélioration

### Description du Projet

Ce projet implémente un système de génération de noms de domaine utilisant des modèles de langage (LLM) avec un focus sur l'évaluation systématique, la découverte de cas limites et l'amélioration itérative.

### Architecture du Système

- **Modèle Générateur**: Llama3.1 via Ollama pour la génération de noms de domaine
- **Système d'Évaluation**: LLM-as-a-Judge pour l'évaluation automatisée
- **API REST**: FastAPI pour le déploiement
- **Garde-fous**: Filtrage de contenu inapproprié

### Installation et Configuration

#### 1. Installation d'Ollama
```bash
# Installation d'Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Téléchargement du modèle Llama3.1
ollama pull llama3.1
```

#### 2. Installation des dépendances Python
```bash
pip install -r requirements.txt
```

#### 3. Configuration des variables d'environnement
```bash
cp .env.example .env
# Éditer .env avec vos clés API si nécessaire
```

### Utilisation


### Structure du Projet

```
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── creation_dataset.py                         # Création de datasets
│   ├── garde_fous.py                               # Système de sécurité
│   ├── generateur_nom_domaine.py                   # Générateur principal
│   ├── evaluateur_llm.py                           # Système d'évaluation
│   └── utils.py                                    # Utilitaires
├── api/
├── data/
│   ├── dataset_experimental.json                   # Dataset généré pour les tests par la classe 
│   └── evaluations/                                # Résultats d'évaluations`CreationDataset`
├── tests/
├── docs/
├── requirements.txt
├── .env.example
└── README.md
```
