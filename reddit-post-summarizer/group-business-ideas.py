import os
import numpy as np
from sklearn.cluster import DBSCAN
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

hf = HuggingFacePipeline.from_model_id(
    model_id="gemma3:latest",
    task="text-generation"
)

template = """Categorize the titles into similar groups in the context of business ideas. 
The ideas should be aimed at IT and software solutions. 
Try to create no more than 10 groups. 
For outliers that don't fit into any group, assign them to a group called 'Miscellaneous'

Given the following titles:
{titles}
."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

persist_dir = "./reddit-chroma-db"
ollama_emb = OllamaEmbeddings(
    model="mxbai-embed-large",
)


def categorize_titles(titles: list[str]) -> str:
    titles_str = "\n".join(f"- {title}" for title in titles)
    result = chain.invoke({"titles": titles_str})
    return result


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["link"] = record.get("link")

    return metadata


if not os.path.exists(persist_dir):
    print("Creating new Chroma DB...")

    loader = JSONLoader(
        file_path="./ideas.json",
        jq_schema=".[]",
        content_key="content",
        metadata_func=metadata_func,
    )
    documents = loader.load()

    db = Chroma.from_documents(
        documents, ollama_emb, persist_directory=persist_dir)
else:
    print("Loading existing Chroma DB...")
    db = Chroma(persist_directory=persist_dir, embedding_function=ollama_emb)

documents_with_scores = db.similarity_search_with_score(query='', k=200)
documents = [doc for doc, score in documents_with_scores]

# Get the IDs of the retrieved documents
document_links = [doc.metadata["link"] for doc in documents]

# Use the underlying Chroma collection to get embeddings
# The `include=['embeddings']` parameter is the key
collection = db.get(include=['embeddings'])
print(collection.keys())

# The returned object has the embeddings
embeddings_list = collection['embeddings']

print(f"Number of embeddings found: {len(embeddings_list)}")

# Convert the list of embeddings to a NumPy array
embeddings_array = np.array(embeddings_list)
print(f"Shape of the NumPy array: {embeddings_array.shape}")
# Define the DBSCAN parameters
eps = 0.7  # Maximum distance between points in a cluster
min_samples = 2  # Minimum number of points required to form a cluster

db_scan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the DBSCAN model to the document embeddings
db_scan.fit(embeddings_array)

# Get the cluster labels for each document
cluster_labels = db_scan.labels_

print(cluster_labels)

# Create a dictionary to store the grouped documents
grouped_documents = {}

# Iterate over the documents and their cluster labels
for document, label in zip(documents, cluster_labels):
    if label not in grouped_documents:
        grouped_documents[label] = []
    grouped_documents[label].append(document)


# Print the grouped documents
for label, documents in grouped_documents.items():
    print(f"Cluster {label}:")
    for document in documents:
        print(f"\tTitle: {document.metadata['title']}")
        # print(document.page_content)
