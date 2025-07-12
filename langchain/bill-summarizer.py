import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
import re

ollama_emb = OllamaEmbeddings(
    model="mxbai-embed-large",
)

chat = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3.2",
    temperature = 0.8,
    num_predict = 512,
)

def chunk_text_with_semantic(document_path, ollama_embeddings_model):
    documents = []

    with open(document_path, 'r', encoding='utf-8') as file:
        full_document_content = file.read()

    text_splitter = SemanticChunker(
        embeddings=ollama_embeddings_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    semantically_chunked_docs = text_splitter.create_documents([full_document_content])

    for i, doc in enumerate(semantically_chunked_docs):
        doc.metadata["source"] = document_path
        doc.metadata["chunk_id"] = i
        documents.append(doc)
    
    return documents

persist_dir = "./chroma_db"  # Your chosen directory name

documents = []
# Check if the database exists
import os
if not os.path.exists(persist_dir):
    print("Creating new Chroma DB...")
    documents = chunk_text_with_semantic('./text-docs/bill.txt', ollama_emb)
    db = Chroma.from_documents(documents, ollama_emb, persist_directory=persist_dir)
    with open("chunked_documents.json", "w", encoding="utf-8") as json_file:
        json.dump(documents, json_file, ensure_ascii=False, indent=2)

else:
    print("Loading existing Chroma DB...")
    db = Chroma(persist_directory=persist_dir, embedding_function=ollama_emb)
    collection = db.get()
    documents = collection["documents"]

messages = [
    (
        "system",
        """You are a helpful assistant that answers questions based on the context and query provided.
        If the knowledge is not in the context, 
        simnply answer that there is not enough information has been supplied to answer the question""",
    )
]

ai_msg = chat.invoke(messages)
print(ai_msg.content)

# Conversation loop for follow-up questions
while True:
    user_input = input("\nAsk a follow-up question (or type 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        print("Exiting conversation.")
        break

    refine_prompt = [
        (
            "system",
            "You are an expert at information retrieval. Given a user's question, pull the relevent keywords to use in a keyword search. only use words/sentences provided in the user's query. return only the keywords separated by commas, without any additional text.",
        ),
        ("human", user_input)
    ]
    refine_response = chat.invoke(refine_prompt)
    refined_query = refine_response.content.strip()
    print(f"Refined similarity search query: {refined_query}")
    keywords = [kw.strip() for kw in refined_query.split(",") if kw.strip()]
    keyword_results = []
    for doc in documents:
        if any(re.search(rf"\b{re.escape(kw)}\b", doc, re.IGNORECASE) for kw in keywords):
            keyword_results.append(doc)

    # Perform similarity search for the new user input
    search_results = db.similarity_search(user_input, k=10)
    simalrity_context = "\n".join([doc.page_content for doc in search_results])
    keyword_context = "\n".join([doc for doc in keyword_results])

    context = simalrity_context + "\n\n" + keyword_context
    print(context)
    messages.append(("human", "Context:" + context + "\n\n Query:\n" + user_input))
    ai_msg = chat.invoke(messages)
    print("----------------------------------------------------------")
    print(ai_msg.content)
    messages.append(("ai", ai_msg.content))
