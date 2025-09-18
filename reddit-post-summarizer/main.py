
from langchain.chains.sequential import SequentialChain
import re
from typing import ClassVar, List, Dict
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from bs4 import BeautifulSoup
import feedparser
import requests
import time
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
import os


subreddits = [
    "https://www.reddit.com/r/Doesthisexist/.rss",
    "https://www.reddit.com/r/apps/.rss",
    "https://www.reddit.com/r/smallbusiness/.rss",
    "https://www.reddit.com/r/SideProject/.rss",
    "https://www.reddit.com/r/SaaS/.rss",
    "https://www.reddit.com/r/Solopreneur/.rss"
]

chat = ChatOllama(
    base_url="http://localhost:11434/",
    model="llama3.2",
    temperature=0.3,
    num_predict=2048,
)

pipe = pipeline("text-classification",
                model="jacobNar/distilbert-5batch-3epoch-reddit-v3", device="cuda")


# -------------------------
# Step 1: FetchRSSChain
# -------------------------

def fetch_feed(url: str, timeout: int = 10, max_retries: int = 5):
    headers = {"User-Agent": "reddit-post-summarizer/0.1 (by /u/yourusername)"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                print(f"429 from {url}, waiting 60s...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            return feedparser.parse(resp.content)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"Request failed ({e}). Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise requests.exceptions.HTTPError(
        f"Failed to fetch {url} after {max_retries} attempts")


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    clean_text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'submitted by.*?comments\]', '', clean_text)
    clean_text = re.sub(r'&#32;', ' ', clean_text)
    return ''.join(c for c in clean_text if ord(c) < 128).strip()


def scan_subreddits(subreddits: List[str]) -> List[Dict]:
    candidates = []
    for feed_url in subreddits:
        print("Fetching", feed_url)
        try:
            feed = fetch_feed(feed_url)
        except Exception as e:
            print(f"Failed {feed_url}: {e}")
            continue
        for entry in feed.entries:
            content = entry.content[0]["value"] if 'content' in entry and entry['content'] else ""
            candidates.append({
                "title": getattr(entry, "title", ""),
                "link": getattr(entry, "link", ""),
                "text": clean_html(content)
            })
    return candidates


class FetchPostsChain(Chain):
    input_keys: ClassVar[List[str]] = []
    output_keys: ClassVar[List[str]] = ["posts"]

    def _call(self, inputs):
        posts = scan_subreddits(subreddits)
        return {"posts": posts}

# -------------------------
# Step 2: Classify Posts
# -------------------------


def call_ollama(prompt: str, max_tokens: int = 2048) -> str:
    try:
        # instantiate ChatOllama client and call .invoke with messages list
        messages = [
            ("system", "You will be passed in a reddit post. Your job is to summarizee it in 2-3 sentence and explain how software can help sove the question askers problem. If software cannot solve the problem, simply output 'I'm not sure'. Don't suggest any specific software or tools. Just explain how software can help solve the problem. Be concise."),
            ("human", prompt),
        ]
        ai_msg = chat.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print("OLLAMA call failed:", e)
        return ""


def classify_post_text_hf(text: str) -> Dict:
    print("classifying")
    result = pipe(text)
    return result[0]


class ClassifyPostsChain(Chain):
    input_keys: ClassVar[List[str]] = ["posts"]
    output_keys: ClassVar[List[str]] = ["classified_posts"]

    def _call(self, inputs):
        posts = inputs["posts"]
        new_posts = []
        print(len(posts), "posts to classify")
        for post in posts:
            try:
                classification = classify_post_text_hf(
                    post["title"] + "\n" + post["text"])

                print(classification['label'],
                      classification['score'], post['title'])
                if (classification['label'] == "positive" and classification['score'] >= 0.50):
                    summary = call_ollama(post["text"])
                    post["summary"] = summary
                    new_posts.append(post)
            except Exception as Excep:
                print(Excep)
                continue

        return {"classified_posts": new_posts}


# classify_chain = ClassifyPostsChain()

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
        "format_instructions": parser.get_format_instructions()},
)


def clean_prompt(prompt: str) -> str:
    return re.sub(r"[^\w\s]", "", prompt)


def categorize_titles(titles: list[str]) -> dict:
    titles_str = "\n".join(f"- {title}" for title in titles)
    clean_input = clean_prompt(titles_str)
    formatted_prompt = prompt.format(titles=clean_input)
    result = chat.invoke(formatted_prompt)
    return parser.parse(result.content)

# -----------------------------
# Step 3 - Categorization Chain
# -----------------------------


class CategorizePostsChain(Chain):
    input_keys: ClassVar[List[str]] = ["classified_posts"]
    output_keys: ClassVar[List[str]] = ["categorized_posts"]

    def _call(self, inputs: dict):
        posts = inputs["classified_posts"]
        titles = [post["title"] for post in posts]

        results = categorize_titles(titles)
        categories = results['categories']

        # Map titles back to posts
        for post in posts:
            post["category"] = next(
                (cat for cat, items in categories.items()
                 if post["title"] in items),
                "Miscellaneous"
            )
        return {"categorized_posts": posts}

# -----------------------------
# Step 4 - Categorization Chain
# -----------------------------


persist_dir = "../reddit-chroma-db"
ollama_emb = OllamaEmbeddings(model="mxbai-embed-large")


class SavePostsChain(Chain):
    input_keys: ClassVar[List[str]] = ["categorized_posts"]
    output_keys: ClassVar[List[str]] = ["saved_count"]

    def _call(self, inputs: dict):
        posts = inputs["categorized_posts"]

        # Convert posts -> Documents, include summary in metadata
        documents = [
            Document(
                page_content=post["text"],
                metadata={
                    "title": post["title"],
                    "link": post["link"],
                    "category": post["category"],
                    "summary": post.get("summary", ""),
                },
            )
            for post in posts
        ]

        if not os.path.exists(persist_dir):
            db = Chroma.from_documents(
                documents, ollama_emb, persist_directory=persist_dir
            )
            saved_count = len(documents)
        else:
            db = Chroma(
                persist_directory=persist_dir, embedding_function=ollama_emb
            )
            new_documents = []
            for doc in documents:
                link = doc.metadata.get("link")
                # Query Chroma for any document with this link in metadata
                # Chroma's API does not support direct metadata filtering, but we can use where argument
                # See: https://docs.trychroma.com/usage-guide#querying
                matches = db.get(where={"link": link}, include=["metadatas"])
                if matches and matches.get("metadatas") and len(matches["metadatas"]) > 0:
                    continue  # Skip if found
                new_documents.append(doc)
            if new_documents:
                db.add_documents(new_documents)
            saved_count = len(new_documents)

        return {"saved_count": saved_count}


fetch_posts_chain = FetchPostsChain()
classify_posts_chain = ClassifyPostsChain()
categorize_posts_chain = CategorizePostsChain()
save_posts_chain = SavePostsChain()

full_chain = SequentialChain(
    chains=[fetch_posts_chain, classify_posts_chain,
            categorize_posts_chain, save_posts_chain],
    input_variables=[],
    output_variables=["saved_count", "posts"]
)


# -------------------------
# Execute
# -------------------------
if __name__ == "__main__":
    result = full_chain({})
    print("Saved posts:", result["saved_count"])
    for post in result["posts"]:
        print(post["title"])
