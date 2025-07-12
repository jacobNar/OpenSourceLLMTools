from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
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


def chunk_text(document_path):
    documents = []
    with open(document_path, 'r', encoding='utf-8') as file:
        document = file.read()
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', document)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # If paragraph is short enough, use as is
        if len(para) <= 512:
            documents.append(Document(page_content=para))
        else:
            # Otherwise, split into sentences and chunk <= 512 chars
            sentences = re.split(r'(?<=[.!?])\s+', para)
            chunk = ""
            for sentence in sentences:
                if len(chunk) + len(sentence) + 1 <= 512:
                    chunk = (chunk + " " + sentence).strip()
                else:
                    if chunk:
                        documents.append(Document(page_content=chunk))
                    chunk = sentence
            if chunk:
                documents.append(Document(page_content=chunk))
    return documents


documents = chunk_text('./text-docs/article.txt')

db = Chroma.from_documents(documents, ollama_emb)

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
    # Perform similarity search for the new user input
    search_results = db.similarity_search(user_input, k=10)
    context = "\n".join([doc.page_content for doc in search_results])
    print("Search Results:")
    print(search_results)
    messages.append(("human", "Context:" + context + "\n\n Query:\n" + user_input))
    ai_msg = chat.invoke(messages)
    print(ai_msg.content)
    messages.append(("ai", ai_msg.content))
