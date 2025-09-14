import os
import re
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader

# -----------------------------
# Setup LLM + Embeddings
# -----------------------------
chat = ChatOllama(
    base_url="http://localhost:11434/",
    model="llama3.2",
    temperature=0.3,
    num_predict=2048,
)

persist_dir = "./reddit-chroma-db"

ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")


# -----------------------------
# Schema + Parser
# -----------------------------
class Categories(BaseModel):
    categories: dict[str, list[str]] = Field(
        description="Mapping of category name to list of titles"
    )


parser = JsonOutputParser(pydantic_object=Categories)

prompt = PromptTemplate(
    template="""
    Categorize the titles into groups of business ideas.
    The ideas should focus on IT and software solutions.
    Try to create no more than 10 groups.
    Outliers should be assigned to 'Miscellaneous'.

    Return ONLY valid JSON following this format:
    {format_instructions}

    Titles:
    {titles}
    """,
    input_variables=["titles"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)


# -----------------------------
# Categorization Function
# -----------------------------
def clean_prompt(prompt):
    return re.sub(r"[^\w\s]", "", prompt)


def categorize_titles(titles: list[str]) -> dict:
    titles_str = "\n".join(f"- {title}" for title in titles)
    clean_input = clean_prompt(titles_str)

    formatted_prompt = prompt.format(titles=clean_input)

    result = chat.invoke(formatted_prompt)

    # Parse and validate JSON
    return parser.parse(result.content)


# -----------------------------
# Metadata Loader
# -----------------------------
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["link"] = record.get("link")
    return metadata


# -----------------------------
# Load or Create DB
# -----------------------------
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

# -----------------------------
# Get documents + categorize
# -----------------------------
documents_with_scores = db.similarity_search_with_score(query="", k=75)
documents = [doc for doc, score in documents_with_scores]

document_titles = [doc.metadata["title"] for doc in documents]

categories = categorize_titles(document_titles)

print(categories['categories'])
