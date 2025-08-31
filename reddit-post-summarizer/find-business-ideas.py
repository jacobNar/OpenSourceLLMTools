from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Dict
import json
import os
from langchain_community.document_loaders import JSONLoader

persist_dir = "./reddit-chroma-db"

ollama_emb = OllamaEmbeddings(
    model="mxbai-embed-large",
)

llm = ChatOllama(
    base_url="http://localhost:11434/",
    model='gemma3:latest',
    temperature=0.2,
    num_predict=512,
    device="cuda"
)


def call_ollama(prompt: str, max_tokens: int = 2048) -> str:
    try:
        # instantiate ChatOllama client and call .invoke with messages list
        messages = [
            ("system", "You will be passed in a reddit post. Your job is to summarizee it in 2-3 sentence and explain how software can help sove the question askers problem. If software cannot solve the problem, simply output 'I'm not sure'. Don't suggest any specific software or tools. Just explain how software can help solve the problem. Be concise."),
            ("human", prompt),
        ]
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print("OLLAMA call failed:", e)
        return ""

# Define the metadata extraction function.


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["link"] = record.get("link")

    return metadata


# if not os.path.exists(persist_dir):
#     print("Creating new Chroma DB...")

#     loader = JSONLoader(
#         file_path="./ideas.json",
#         jq_schema=".[]",
#         content_key="content",
#         metadata_func=metadata_func,
#     )
#     documents = loader.load()

#     db = Chroma.from_documents(
#         documents, ollama_emb, persist_directory=persist_dir)
# else:
#     print("Loading existing Chroma DB...")
#     db = Chroma(persist_directory=persist_dir, embedding_function=ollama_emb)
#     collection = db.get()
#     documents = collection["documents"]


pipe = pipeline("text-classification",
                model="jacobNar/distilbert-5batch-3epoch-reddit-v3", device="cuda")


def classify_post_text_hf(text: str) -> Dict:
    result = pipe(text)
    return result[0]


def main():
    in_path = "posts.json"
    out_path = "ideas.json"
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except FileNotFoundError:
        posts = []

    try:
        with open(out_path, "r", encoding="utf-8") as f:
            ideas = json.load(f)
    except FileNotFoundError:
        ideas = []

    # print(len(posts), "posts loaded from", in_path)
    existing_links = set(item["link"] for item in ideas if "link" in item)

    added = 0
    for post in posts:
        try:
            classification = classify_post_text_hf(
                post["title"] + "\n" + post["content"])
        except Exception as e:
            print(
                f"Failed to classify post {post['title']} {post['link']}: {e}")
            continue
        print(classification)
        if (classification['label'] == "positive" and classification['score'] >= 0.50):

            if post["link"] not in existing_links:
                summary = call_ollama(post["content"])
                post["summary"] = summary
                ideas.append(post)
                existing_links.add(post["link"])
                added += 1
            else:
                print(
                    f"Skipping existing post: {post['title']} {post['link']}. Classification: {classification}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ideas, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(ideas)} items saved to {out_path}")
    for i, item in enumerate(ideas, 1):
        print(i, item["title"], item["link"])


if __name__ == "__main__":
    main()
