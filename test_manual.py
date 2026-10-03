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