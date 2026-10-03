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
# TEST MANUEL
# ============================================================

if __name__ == "__main__":
    print("=== Test du chargement et du chunking ===\n")
    
    # 1. Charger le corpus
    documents = load_corpus()
    print(f"📚 {len(documents)} documents chargés")
    
    # 2. Afficher les sources
    print("\nSources :")
    for doc in documents[:5]:
        print(f"  - {doc.metadata['source']}")
    print(f"  ... et {len(documents) - 5} autres")
    
    # 3. Découper en chunks
    chunks = split_documents(documents)
    print(f"\n✂️  {len(chunks)} chunks créés")
    
    # 4. Afficher un exemple de chunk
    print("\nExemple de chunk :")
    print("─" * 60)
    print(chunks[0].page_content[:300])
    print("─" * 60)
    print(f"Source : {chunks[0].metadata['source']}")