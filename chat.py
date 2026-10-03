"""
Script de chat pour l'assistant RAG Randoneo.

Au lancement :
1. Charge et indexe le corpus
2. Ouvre une boucle interactive
3. Affiche la réponse + les sources
4. Quitte sur "quit"
"""

from rag import (
    load_corpus,
    split_documents,
    build_index,
    build_retriever,
    build_rag_chain,
)


def main():
    """Point d'entrée du chat."""
    
    # ============================================================
    # 1. CONSTRUCTION DE L'INDEX (une seule fois)
    # ============================================================
    
    print("=" * 60)
    print("  Assistant RAG Randoneo")
    print("=" * 60)
    print()
    
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
    
    print()
    print("=" * 60)
    print("  Prêt ! Posez vos questions (tapez 'quit' pour quitter)")
    print("=" * 60)
    print()
    
    # ============================================================
    # 2. BOUCLE INTERACTIVE
    # ============================================================
    
    while True:
        try:
            # Lit la question au clavier
            question = input("❓ Votre question : ").strip()
        except (EOFError, KeyboardInterrupt):
            # Gère Ctrl+D et Ctrl+C
            print("\n\n👋 À bientôt !")
            break
        
        # Détecte la commande de sortie
        if question.lower() in ("quit", "exit", "q"):
            print("👋 À bientôt !")
            break
        
        # Ignore les lignes vides
        if not question:
            continue
        
        # ============================================================
        # 3. EXÉCUTION DU RAG
        # ============================================================
        
        print()
        
        try:
            # Lance la chaîne RAG
            result = rag_chain.invoke(question)
            
            # Affiche la réponse
            print("💬 Réponse :")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)
            
            # Affiche les sources (uniques et triées)
            sources = sorted({doc.metadata["source"] for doc in result["context"]})
            print(f"\n📎 Sources ({len(sources)}) :")
            for source in sources:
                print(f"   - {source}")
            
            print()
        
        except Exception as e:
            print(f"❌ Erreur : {e}")
            print()


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()