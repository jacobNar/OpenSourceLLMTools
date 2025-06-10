from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
import os
from langchain_community.document_loaders import UnstructuredXMLLoader
import xml.etree.ElementTree as ET

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

def chunk_text(document_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
    )

    with open(document_path, 'r', encoding='utf-8') as file:
        document = file.read()

    sentences = text_splitter.split_text(document)
    documents = []
    for sentence in sentences:
        documents.append(Document(page_content=sentence))
    return documents

def chunk_xml(document_path):
    documents = []
    tree = ET.parse(document_path)
    root = tree.getroot()

    legis_body = root.find('.//legis-body')

    if legis_body is not None:
        for section in legis_body.findall('.//section'):
            section_id = section.get('id')
            print(f"Processing section with ID: {section_id}")
            section_header_elem = section.find('header')
            print(f"Section header element: {section_header_elem}")
            section_header = section_header_elem.text.strip() if section_header_elem is not None and section_header_elem.text else ""
            # Get all text in section
            section_text = ' '.join(section.itertext()).strip()
            # Check for subsections if text is long
            subsections = section.findall('.//subsection')
            if len(section_text) > 1000 and subsections:
                for subsection in subsections:
                    subsection_header_elem = subsection.find('header')
                    subsection_header = subsection_header_elem.text.strip() if subsection_header_elem is not None and subsection_header_elem.text else ""
                    print(f"Processing subsection with header: {subsection_header}")
                    subsection_text = ' '.join(subsection.itertext()).strip()
                    document = Document(
                        page_content=subsection_text,
                        metadata={
                            "section-id": section_id,
                            "header": subsection_header
                        }
                    )
                    documents.append(document)
            else:
                document = Document(
                    page_content=section_text,
                    metadata={
                        "section-id": section_id,
                        "header": section_header
                    }
                )
                documents.append(document)
    return documents

# Load and chunk the text document
# text_documents = chunk_text('./text-docs/article.txt')

# Example usage for XML document
xml_documents = chunk_xml('../rag-docs/BILLS-119hr1eh.xml')

# vector_store.add_documents(documents=text_documents)
# vector_store.add_documents(documents=xml_documents)

for doc in xml_documents:
    try:
        print(f"Embedding document: {doc.metadata['header']}")
        vector_store.add_documents([doc])  # Add documents one by one
    except Exception as e:
        print(f"Error embedding document: {doc.metadata['header']}")
        print(f"Error: {e}")

query = "how will the health insurance indsutry be affected by this bill."
similarity_search_query = "health, insurance, medicaid, medicare  "

search_results = vector_store.similarity_search(query=similarity_search_query,k=10)
print("Search Results:")
print(search_results)
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