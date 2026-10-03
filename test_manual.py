# ============================================================
# TEST MANUEL fichier rag.py
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