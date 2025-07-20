import json
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any

ollama_emb = OllamaEmbeddings(
    model="mxbai-embed-large",
)

chat = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3.2",
    temperature = 0.8,
    num_predict = 512,
)

def rerank_documents_hf(
    query: str,
    documents: List[Document],
    model_name: str = "BAAI/bge-reranker-large",
    top_k: int = 10
) -> List[Dict[str, Any]]:
    if not documents:
        return []

    # Extract page content from Document objects
    document_texts = [doc.page_content for doc in documents]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pairs = [[query, doc] for doc in document_texts]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        scores = model(**inputs).logits.squeeze().tolist()

    scored_documents = []
    for i, doc in enumerate(documents):
        scored_documents.append({"context": doc.page_content, "score": scores[i], "chunk_id": doc.metadata['chunk_id']})

    scored_documents.sort(key=lambda x: x["score"], reverse=True)
    top_scored_documents = scored_documents[:top_k]

    # Extract the original documents based on the sorted scores
    reranked_documents = [
        next(doc for doc in documents if doc.page_content == item["context"])
        for item in top_scored_documents
    ]

    return reranked_documents


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
        The context will be a collection of excerpts taken from a bill document.
        If the knowledge is in the context, answer the question based on the context.
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
    print(f"Refined keyword search query: {refined_query}")
    keywords = [kw.strip() for kw in refined_query.split(",") if kw.strip()]
    keyword_results = []
    
    # Ensure 'documents' and 'metadatas' keys exist
    if 'documents' in collection and 'metadatas' in collection:
        documents_content = collection['documents']
        documents_metadata = collection['metadatas']

        # Iterate through the documents and their metadata
        for i, doc_content in enumerate(documents_content):
            if any(re.search(rf"\b{re.escape(kw)}\b", doc_content, re.IGNORECASE) for kw in keywords):
                # Create a Document object with content and metadata
                doc = Document(page_content=doc_content, metadata=documents_metadata[i])
                keyword_results.append(doc)
    else:
        print("Error: 'documents' or 'metadatas' key missing in Chroma DB retrieval.")
    # keyword_ranked_results = rerank_documents_hf(user_input, [doc.page_content for doc in keyword_results])

    # Perform similarity search for the new user input
    search_results = db.similarity_search(user_input, k=10)
    simalrity_context = "\n".join([doc.page_content for doc in search_results])
    # search_ranked_results = rerank_documents_hf(user_input, [doc.page_content for doc in search_results])

    combined_results = search_results + keyword_results

    seen = set()
    unique_results = []
    for doc in combined_results:
        identifier = doc.page_content  # or use another unique field if available
        if identifier not in seen:
            unique_results.append(doc)
            seen.add(identifier)

    reranked_results = rerank_documents_hf(user_input, unique_results)
    # print(reranked_results)
    top_k_results_with_siblings = reranked_results[:3]
    final_context = []
    for i, result in enumerate(top_k_results_with_siblings):
        current_chunk_id = result.metadata['chunk_id']
        previous_chunk_id = current_chunk_id - 1
        next_chunk_id = current_chunk_id + 1
        print(f"Processing Result {i} with chunk_id: {current_chunk_id}, previous_chunk_id: {previous_chunk_id}, next_chunk_id: {next_chunk_id}")
        # Fetch the previous, current, and next chunks
        previous_chunks = db.get(where={"chunk_id": previous_chunk_id})['documents']
        current_chunks = db.get(where={"chunk_id": current_chunk_id})['documents']
        next_chunks = db.get(where={"chunk_id": next_chunk_id})['documents']
        # print(previous_chunks[0], current_chunks[0], next_chunks[0])

        # Add the chunks to final_context in order
        if previous_chunks:
            final_context.extend(previous_chunks)
        if current_chunks:
            final_context.extend(current_chunks)
        if next_chunks:
            final_context.extend(next_chunks)
            
    context = ""
    for i, result in enumerate(final_context, 0):
        context += f"Result {i}\n{result}\n\n"
    print(context)
    
    messages.append(("human", "Context:" + context + "\n\n Query:\n" + user_input))
    ai_msg = chat.invoke(messages)
    print("----------------------------------------------------------")
    print(ai_msg.content)
    messages.append(("ai", ai_msg.content))
