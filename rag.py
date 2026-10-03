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
# TEST MANUEL
# ============================================================

if __name__ == "__main__":
    print("=== Test du chargement, du chunking et de l'indexation ===\n")
    
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
    
    # 5. Tester une recherche
    print("\n🔍 Test de recherche...")
    question = "Combien pèse la tente Aero 2 places ?"
    print(f"Question : {question}\n")
    
    results = retriever.invoke(question)
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.metadata['source']}")
        print(f"      {doc.page_content[:80]}...")
        print()