# cli.py
"""
CLI interface for testing ContextEngine locally.
Run from the project root: python cli.py
"""

from generation.chat import DocumentationBot

bot = DocumentationBot()

print("\n========================================")
print("  ContextEngine — Local CLI Test")
print("========================================")
print("  Type your question and press Enter.")
print("  Commands:  'reset' | 'history' | 'exit'")
print("========================================\n")

while True:
    try:
        query = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting. Goodbye!")
        break

    if not query:
        continue

    if query.lower() == "exit":
        print("Exiting. Goodbye!")
        break

    if query.lower() == "reset":
        bot.reset()
        print("--- Conversation history cleared ---\n")
        continue

    if query.lower() == "history":
        if not bot.conversation_history:
            print("--- No history yet ---\n")
        else:
            print("--- Conversation History ---")
            for msg in bot.conversation_history:
                label = "You" if msg["role"] == "user" else "Bot"
                print(f"{label}: {msg['content'][:120]}...")
            print(f"--- {bot.history_turns} turn(s) in memory ---\n")
        continue
    if query.lower() == "debug":
        from retrieval.retriever import retrieve_context
        chunks = retrieve_context(
            input("Query to debug: "),
            user_role=st.session_state.get("role", "admin")
        )
        print(f"\n--- {len(chunks)} chunks retrieved ---")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] score={c['similarity_score']}  file={c.get('file_path','?')}")
            print(c["text"][:300])
        print("---\n")
        continue
    if query.lower() == "paths":
        import chromadb
        from config.config import settings

        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        col    = client.get_collection(settings.chroma_collection_name)
        all_docs = col.get(include=["metadatas"])

        # Collect unique file paths
        paths = sorted(set(
            m.get("file_path") or m.get("file_name") or "unknown"
            for m in all_docs["metadatas"]
        ))

        print(f"\n--- {len(paths)} unique paths in ChromaDB ---")
        for p in paths:
            print(f"  {p}")
        print()
        continue
    response = bot.ask(query)
    print(f"\nContextEngine: {response}\n")