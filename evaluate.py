"""
Script d'évaluation du RAG documentaire Randoneo.

Il rejoue le RAG sur le jeu de questions (eval/rag_questions.jsonl)
et calcule les 4 métriques RAGAS :
- context_precision
- context_recall
- faithfulness
- answer_relevancy

Les scores sont moyennés sur les questions in_corpus.
Les questions out_of_corpus sont jugées sur leur comportement de refus.
"""

import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from rag import (
    build_index,
    build_rag_chain,
    build_retriever,
    load_corpus,
    split_documents,
)


# ============================================================
# 1. CHARGEMENT DES QUESTIONS
# ============================================================

def load_questions(path: str = "eval/rag_questions.jsonl") -> list[dict]:
    """
    Charge les questions depuis un fichier JSONL.
    
    Args:
        path: Chemin vers le fichier JSONL
    
    Returns:
        Liste de dictionnaires avec question, ground_truth, source_doc, scope
    """
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


# ============================================================
# 2. CONSTRUCTION DU RAG
# ============================================================

def build_rag():
    """Construit le RAG (index + chaîne)."""
    print("📚 Chargement du corpus...")
    documents = load_corpus()
    print(f"   {len(documents)} documents chargés")
    
    print("✂️  Découpage en chunks...")
    chunks = split_documents(documents)
    print(f"   {len(chunks)} chunks créés")
    
    print("🔨 Indexation...")
    vectorstore = build_index(chunks)
    print(f"   {vectorstore._collection.count()} vecteurs indexés")
    
    print("🔗 Assemblage de la chaîne RAG...")
    retriever = build_retriever(vectorstore, k=4)
    rag_chain = build_rag_chain(retriever)
    
    return retriever, rag_chain


# ============================================================
# 3. PRÉPARATION DES ÉCHANTILLONS RAGAS
# ============================================================

def build_samples(
    questions: list[dict],
    retriever,
    rag_chain,
) -> list[SingleTurnSample]:
    """
    Rejoue le RAG sur chaque question et construit les échantillons RAGAS.
    
    Args:
        questions: Liste de questions
        retriever: Le retriever
        rag_chain: La chaîne RAG
    
    Returns:
        Liste de SingleTurnSample
    """
    samples = []
    for i, item in enumerate(questions, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"   [{i}/{len(questions)}] {question[:60]}...")
        
        # Récupère les passages
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # Génère la réponse
        result = rag_chain.invoke(question)
        answer = result["answer"]
        
        # Crée l'échantillon RAGAS
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=ground_truth,
            )
        )
    
    return samples


# ============================================================
# 4. ÉVALUATION
# ============================================================

def run_evaluation(samples: list[SingleTurnSample]):
    """
    Lance RAGAS sur les échantillons.
    
    Args:
        samples: Liste de SingleTurnSample
    
    Returns:
        Le DataFrame RAGAS
    """
    # Crée le juge (Claude)
    judge = LangchainLLMWrapper(
        ChatAnthropic(model="claude-haiku-4-5", max_tokens=2048)
    )
    
    # Crée les embeddings du juge (OpenAI)
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    
    # Lance l'évaluation
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=judge,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1),
    )
    
    return result


# ============================================================
# 5. AFFICHAGE
# ============================================================

def display_results(df, scope_filter: str = "in_corpus"):
    """
    Affiche les scores moyens et le détail.
    
    Args:
        df: DataFrame RAGAS
        scope_filter: 'in_corpus' ou 'out_of_corpus'
    """
    metrics = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    
    print()
    print("=" * 60)
    print(f"  Scores moyens ({scope_filter})")
    print("=" * 60)
    
    for metric in metrics:
        if metric in df.columns:
            score = df[metric].mean()
            print(f"  {metric:20s} : {score:.3f}")
    
    print()
    print("=" * 60)
    print(f"  Détail par question ({scope_filter})")
    print("=" * 60)
    
    for i, row in df.iterrows():
        print(f"\n  Q{i+1} : {row['user_input'][:70]}")
        for metric in metrics:
            if metric in df.columns:
                value = row[metric]
                if value is not None:
                    print(f"         {metric:20s} = {value:.3f}")


# ============================================================
# 6. POINT D'ENTRÉE
# ============================================================

def main():
    """Point d'entrée du script d'évaluation."""
    
    # Charge le .env
    load_dotenv()
    
    # Vérifie les clés
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY manquante dans .env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY manquante dans .env")
    
    print("=" * 60)
    print("  Évaluation du RAG Randoneo")
    print("=" * 60)
    print()
    
    # 1. Construit le RAG
    retriever, rag_chain = build_rag()
    print()
    
    # 2. Charge les questions
    print("📋 Chargement des questions...")
    questions = load_questions()
    in_corpus = [q for q in questions if q["scope"] == "in_corpus"]
    out_of_corpus = [q for q in questions if q["scope"] == "out_of_corpus"]
    print(f"   {len(questions)} questions chargées")
    print(f"   - {len(in_corpus)} in_corpus")
    print(f"   - {len(out_of_corpus)} out_of_corpus")
    print()
    
    # 3. Évalue les in_corpus
    print("🔍 Évaluation (in_corpus)...")
    samples = build_samples(in_corpus, retriever, rag_chain)
    print()
    print("⏳ Calcul des métriques RAGAS (peut prendre 1-2 minutes)...")
    result = run_evaluation(samples)
    df = result.to_pandas()
    
    # 4. Affiche les résultats
    display_results(df, "in_corpus")
    
    # 5. Juge les out_of_corpus
    if out_of_corpus:
        print()
        print("=" * 60)
        print("  Questions hors corpus (out_of_corpus)")
        print("=" * 60)
        
        for item in out_of_corpus:
            question = item["question"]
            result = rag_chain.invoke(question)
            answer = result["answer"]
            
            print(f"\n  Q : {question}")
            print(f"  R : {answer}")
            print(f"  → Comportement : {'✅ Refus' if 'pas' in answer.lower() or 'dispose pas' in answer.lower() else '⚠️ À vérifier'}")


if __name__ == "__main__":
    main()