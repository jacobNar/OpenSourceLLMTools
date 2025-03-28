from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
import os

embed = OllamaEmbeddings(
    model="llama3.2"
)

chat = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3.2",
    temperature = 0.8,
    num_predict = 256,
)

vector_store = InMemoryVectorStore(embed)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

with open('./text-docs/article.txt', 'r', encoding='utf-8') as file:
    document = file.read()

sentences = text_splitter.split_text(document)
documents = []
for sentence in sentences:
    documents.append(Document(page_content=sentence))

vector_store.add_documents(documents=documents)

query = "what car companies are mentioned in the article?"
query_embedding = embed.embed_query(query)

search_results = vector_store.similarity_search(query=query,k=10)
context = "\n".join([doc.page_content for doc in search_results])
messages = [
    (
        "system",
        """You are a helpful assistant that answers questions based on the context and query provided.
        If the knowledge is not in the context, 
        simnply answer that there is not enough information has been supplied to answer the question""",
    ),
    ("human", "Context:" + context + "\n\n Query:\n" + query),
]

ai_msg = chat.invoke(messages)
print(ai_msg.content)