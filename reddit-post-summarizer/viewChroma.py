from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

persist_dir = "../reddit-chroma-db"
ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")


def main():
    db = Chroma(persist_directory=persist_dir, embedding_function=ollama_emb)
    results = db.get(include=["documents", "metadatas"])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"Row {i+1}:")
        print("  Title:", meta.get("title", "<no title>"))
        print("  Link:", meta.get("link", "<no link>"))
        print("  Category:", meta.get("category", "<no category>"))
        print("  Target:", meta.get("target", "<no target>"))
        print("  Text:", doc[:200] + ("..." if len(doc) > 200 else ""))
        print("  Metadata:", meta)
        print("-"*60)


if __name__ == "__main__":
    main()
