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