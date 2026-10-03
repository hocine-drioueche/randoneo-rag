# RAG Documentaire Randoneo

Un RAG documentaire évalué pour Randoneo, une boutique d'équipement outdoor.
Le système répond aux questions des clients en s'appuyant sur le corpus
documentaire, en citant ses sources, et refuse d'inventer quand l'information
n'est pas dans la documentation.

## Statut du projet

✅ **Fonctionnel** — RAG documentaire évalué
- Context precision : **0.89**
- Context recall : **1.00**
- Faithfulness : **0.81**
- Answer relevancy : **0.63**

## Fonctionnalités

- **Indexation** : charge le corpus Markdown, le découpe en chunks, le vectorise
- **Retrieval** : récupère les 4 passages les plus proches d'une question
- **Génération ancrée** : répond en s'appuyant UNIQUEMENT sur le contexte
- **Sources citées** : affiche les documents d'origine
- **Garde-fou** : refuse d'inventer quand l'information n'est pas dans le corpus
- **Évaluation** : mesure la qualité avec RAGAS (4 métriques)

## Architecture

```
randoneo-rag/
├── .env                    # Clés API (non commité)
├── .env.example            # Modèle
├── .gitignore              # Fichiers ignorés
├── README.md               # Documentation
├── requirements.txt        # Dépendances
├── rag.py                  # Module partagé (pipeline RAG)
├── chat.py                 # Script de chat interactif
├── evaluate.py             # Script d'évaluation RAGAS
├── corpus/                 # Documentation Randoneo
│   ├── produits/           # Fiches produit
│   ├── guides/             # Guides pratiques
│   ├── politiques/         # Politiques (retour, garantie...)
│   └── faq/                # FAQ
└── eval/                   # Jeux de questions
    ├── rag_questions.jsonl
    └── rag_questions_thematiques.jsonl
```

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/hocine-drioueche/randoneo-rag.git
cd randoneo-rag
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les clés API

Crée un fichier `.env` à la racine du projet :

```bash
cp .env.example .env
nano .env
```

**Contenu du `.env`** :

```
OPENAI_API_KEY=sk-proj-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### 5. Corriger le stub RAGAS (IMPORTANT)

**RAGAS 0.4.3** importe un module qui n'existe plus dans les versions récentes
de `langchain-community`. Il faut créer un **stub** :

```bash
# Trouver le chemin
python -c "import langchain_community.chat_models; print(langchain_community.chat_models.__file__)"
```

**Puis créer le stub** (remplace le chemin par le tien) :

```bash
cat > .venv/lib/python3.12/site-packages/langchain_community/chat_models/vertexai.py << 'ENDOFFILE'
"""
Stub pour compatibilité RAGAS.
"""
class ChatVertexAI:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Utilisez langchain-google-vertexai")
ENDOFFILE
```

**Pourquoi ?** Parce que RAGAS importe `langchain_community.chat_models.vertexai`
qui a été déplacé dans un package séparé.

## Utilisation

### Chat interactif

```bash
python chat.py
```

**Exemple de session** :

```
============================================================
  Assistant RAG Randoneo
============================================================

📚 Chargement du corpus...
   20 documents chargés
✂️  Découpage en chunks...
   64 chunks créés
🔨 Indexation...
   64 vecteurs indexés
🔗 Assemblage de la chaîne RAG...

============================================================
  Prêt ! Posez vos questions (tapez 'quit' pour quitter)
============================================================

❓ Votre question : Combien pèse la tente Aero 2 places ?

💬 Réponse :
------------------------------------------------------------
La Tente Aero 2 places pèse 1,9 kg (1900 g)...
------------------------------------------------------------

📎 Sources (3) :
   - guides/choisir-sa-tente.md
   - produits/tente-aero-2p.md
   - produits/tente-dome-3p.md
```

### Évaluation

```bash
python evaluate.py
```

**Exemple de sortie** :

```
============================================================
  Scores moyens (in_corpus)
============================================================
  context_precision    : 0.890
  context_recall       : 1.000
  faithfulness         : 0.807
  answer_relevancy     : 0.633

============================================================
  Questions hors corpus (out_of_corpus)
============================================================
  Q : Proposez-vous un programme de fidélité ?
  R : Je n'ai pas d'information sur un programme de fidélité...
  → Comportement : ✅ Refus
```

## Comment ça marche

### Le pipeline RAG

```
Phase d'indexation (une fois) :
Documents → Chunks → Embeddings → Chroma

Phase de requête (à chaque question) :
Question → Embedding → Retrieval → Prompt augmenté → LLM → Réponse ancrée
```

### Les 4 étapes

| Étape | Rôle |
|-------|------|
| **1. Chargement** | Lit les fichiers Markdown du corpus |
| **2. Chunking** | Découpe en chunks de 600 caractères (overlap 100) |
| **3. Indexation** | Vectorise avec OpenAI, stocke dans Chroma |
| **4. Génération** | Récupère les 4 passages, génère avec Claude |

### Les 4 métriques

| Métrique | Côté | Score obtenu |
|----------|------|--------------|
| **Context precision** | Récupération | 0.89 |
| **Context recall** | Récupération | 1.00 |
| **Faithfulness** | Génération | 0.81 |
| **Answer relevancy** | Génération | 0.63 |

## Compétences mobilisées

- **Chunking** : découpage du corpus (`RecursiveCharacterTextSplitter`)
- **Embeddings** : vectorisation avec OpenAI (`text-embedding-3-small`)
- **Base vectorielle** : indexation Chroma
- **RAG** : retrieval + génération ancrée
- **Évaluation** : RAGAS (4 métriques)
- **Ingénierie** : projet Python structuré et reproductible

## Choix techniques

| Choix | Pourquoi |
|-------|----------|
| **Chroma en mémoire** | Simple, pas d'infra |
| **`text-embedding-3-small`** | Rapide, performant, pas cher |
| **Claude Haiku** | Économique, rapide |
| **`temperature=0`** | Réponses fiables |
| **`k=4`** | Bon compromis |
| **Chunks de 600** | Contexte suffisant |

## Auteur

Hocine Drioueche — [drioueche.hocine@gmail.com](mailto:drioueche.hocine@gmail.com)
