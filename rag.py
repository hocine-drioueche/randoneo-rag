"""
Module partagé pour le RAG documentaire Randoneo.

Ce module contient :
- Le chargement du corpus
- Le découpage en chunks
- La construction de l'index vectoriel
- La chaîne RAG
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)


# ============================================================
# 1. CHARGEMENT DU CORPUS
# ============================================================

def load_corpus(corpus_dir: str = "corpus") -> list[Document]:
    """
    Charge tous les fichiers .md du dossier corpus.
    
    Args:
        corpus_dir: Le dossier contenant le corpus
    
    Returns:
        Une liste de Document LangChain, avec le chemin source en métadonnée
    """
    corpus_path = Path(corpus_dir)
    documents = []
    
    # Parcours récursif de tous les fichiers .md
    for file_path in sorted(corpus_path.rglob("*.md")):
        # Chemin relatif (ex: "produits/tente-aero-2p.md")
        relative_path = file_path.relative_to(corpus_path)
        
        # Lit le contenu
        content = file_path.read_text(encoding="utf-8")
        
        # Crée un Document avec la source en métadonnée
        documents.append(
            Document(
                page_content=content,
                metadata={"source": str(relative_path)},
            )
        )
    
    return documents


# ============================================================
# 2. DÉCOUPAGE EN CHUNKS
# ============================================================

def split_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Découpe les documents en chunks avec RecursiveCharacterTextSplitter.
    
    Args:
        documents: La liste de Document
        chunk_size: Taille max d'un chunk (en caractères)
        chunk_overlap: Recouvrement entre chunks
    
    Returns:
        Une liste de chunks (Document)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",   # Titres de section (H2)
            "\n### ",  # Sous-titres (H3)
            "\n\n",    # Paragraphes
            "\n",      # Lignes
            ". ",      # Phrases
            " ",       # Mots
            "",        # Caractères
        ],
    )
    
    chunks = splitter.split_documents(documents)
    return chunks

# ============================================================
# 3. CONSTRUCTION DE L'INDEX VECTORIEL
# ============================================================

# Charge le fichier .env (contient les clés API)
load_dotenv()

# Vérifie que la clé OpenAI est présente
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY manquante. "
        "Vérifie ton fichier .env à la racine du projet."
    )

# Crée la fonction d'embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def build_index(chunks: list[Document]) -> Chroma:
    """
    Vectorise les chunks et les indexe dans Chroma.
    
    Args:
        chunks: La liste de chunks
    
    Returns:
        Le vectorstore Chroma
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vectorstore


def build_retriever(vectorstore: Chroma, k: int = 4):
    """
    Crée un retriever à partir du vectorstore.
    
    Args:
        vectorstore: Le vectorstore Chroma
        k: Nombre de passages à récupérer
    
    Returns:
        Le retriever LangChain
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever



# ============================================================
# 4. LA CHAÎNE RAG
# ============================================================

# Vérifie que la clé Anthropic est présente
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError(
        "ANTHROPIC_API_KEY manquante. "
        "Vérifie ton fichier .env à la racine du projet."
    )

# Crée le modèle Claude
model = init_chat_model(
    "claude-haiku-4-5",
    model_provider="anthropic",
    temperature=0,
    max_retries=8,
)

# Le prompt strict (anti-hallucination)
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu es l'assistant de support de Randoneo, un e-commerce de matériel outdoor. "
        "Réponds en français, avec un ton chaleureux et professionnel.\n\n"
        "RÈGLES STRICTES :\n"
        "1. Appuie-toi UNIQUEMENT sur le contexte fourni ci-dessous.\n"
        "2. N'invente JAMAIS une information absente du contexte.\n"
        "3. Si le contexte ne contient pas la réponse, dis clairement "
        "que tu ne disposes pas de cette information.\n"
        "4. Sois concis : 3 à 5 phrases maximum.\n"
        "5. Termine par une prochaine étape utile si pertinent.",
    ),
    (
        "human",
        "Contexte :\n{context}\n\nQuestion : {question}",
    ),
])


def format_docs(docs) -> str:
    """
    Formate une liste de Document en texte lisible.
    
    Args:
        docs: Liste de Document
    
    Returns:
        Le texte formaté avec les sources
    """
    return "\n\n".join(
        f"[{doc.metadata['source']}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(retriever):
    """
    Assemble la chaîne RAG complète.
    
    La chaîne :
    1. Récupère les passages (retriever)
    2. Formate le contexte (format_docs)
    3. Génère la réponse (prompt + model)
    
    Args:
        retriever: Le retriever LangChain
    
    Returns:
        La chaîne LCEL
    """
    # Sous-chaîne de génération
    generate = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | RAG_PROMPT
        | model
        | StrOutputParser()
    )
    
    # Chaîne complète : récupère + génère
    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough(),
        })
        .assign(answer=generate)
    )
    
    return rag_chain









    
# ============================================================
# TEST MANUEL
# ============================================================

if __name__ == "__main__":
    print("=== Test de la chaîne RAG complète ===\n")
    
    # 1. Charger le corpus
    documents = load_corpus()
    print(f"📚 {len(documents)} documents chargés")
    
    # 2. Découper en chunks
    chunks = split_documents(documents)
    print(f"✂️  {len(chunks)} chunks créés")
    
    # 3. Construire l'index
    print("\n🔨 Construction de l'index...")
    vectorstore = build_index(chunks)
    print(f"✅ Index construit ({vectorstore._collection.count()} vecteurs)")
    
    # 4. Créer le retriever
    retriever = build_retriever(vectorstore, k=4)
    
    # 5. Créer la chaîne RAG
    print("\n🔗 Assemblage de la chaîne RAG...")
    rag_chain = build_rag_chain(retriever)
    print("✅ Chaîne RAG prête\n")
    
    # 6. Tester avec des questions
    questions = [
        "Combien pèse la tente Aero 2 places ?",
        "Sous combien de jours suis-je remboursé après un retour ?",
        "Proposez-vous une carte de fidélité ?",
    ]
    
    for question in questions:
        print("=" * 60)
        print(f"❓ {question}")
        print("=" * 60)
        
        result = rag_chain.invoke(question)
        
        print(f"\n💬 Réponse :\n{result['answer']}\n")
        
        # Afficher les sources uniques
        sources = sorted({doc.metadata["source"] for doc in result["context"]})
        print(f"📎 Sources : {', '.join(sources)}\n")