import json
from bs4 import BeautifulSoup
import re
from typing import List, Dict
from langchain_ollama import ChatOllama
from transformers import pipeline
import requests
import time
import feedparser

OLLAMA_MODEL = "llama3.2"


def call_ollama(prompt: str, max_tokens: int = 2048) -> str:
    try:
        # instantiate ChatOllama client and call .invoke with messages list
        llm = ChatOllama(
            base_url="http://localhost:11434/",
            model=OLLAMA_MODEL,
            temperature=0.2,
            num_predict=max_tokens,
            device="cuda"
        )
        messages = [
            ("system", "You are a helpful assistant. Answer succinctly and output only the assistant text."),
            ("human", prompt),
        ]
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print("OLLAMA call failed:", e)
        return ""


CLASSIFY_PROMPT = """You are an assistant. Given the following Reddit post content summary, decide whether the post is s a request for a software, website, or app solution. Or if the post is about a problem that could be solved by software. Mark "match" as True if:
- a user asking for a recommendation for software, a website, or an app.
- asking for an improved software/tool compared to what they use.
- or describing a business pain point that could be solved by software.

Mark "match" as False if:
- the poster is asking for peoplet to check out something they made such as an app, website, or tool.
- The post is discussing a tool, website, or app that has already been built, regardless of who created it (e.g., sharing a new app, asking for feedback on an app, promoting a product).
- The post is not related to software, websites, or apps.
- The post is a general discussion about a problem without explicitly seeking a software solution.
- The post is asking for feedback on an idea the poster has already built.

Here are some examples of some posts that should be marked as "match": False:
- A user has built an app that can help track people's sleep habits, it can accessed here: https://example.com
- The poster is looking feedback on a website they built to help people find local events near them.
- A user has created a mobile app called JigSwap...

DO NOT mark "match" as True if the post is about a product that has already been built, or asking for feedback. If the words 'built', 'launched', 'created', 'made', 'developed', or similar appear in the post, it is likely about something that already exists.

Answer in strict JSON with keys:
{{"match": true|false, "category": "saas|improve|pain|other", "reason": "<one-line reason>"}}

Post content:
---
{content}
---
Only output valid JSON.
"""

COMMENTS_PROMPT = """You are an assistant. The following are comments from a Reddit post. Decide whether any comment provides a clear, actionable software solution that answers the original poster's question or pain point.
Answer in JSON exactly: {{"found": true|false, "example": "<short example or empty>"}}
Comments:
---
{comments}
---
Only output valid JSON.
"""

HF_CLASSIFY_PROMPT = (
    "Given the following Reddit post summary, decide whether the post is:\n"
    "- a user asking for a recommendation for software, a website, or an app.\n"
    "- asking for an improved software/tool compared to what they use.\n"
    "- or describing a business pain point that could be solved by software.\n\n"
    "If the post is a request for a software, website, or app solution, whether it's for a new recommendation or an improvement on an existing one, mark 'match' as True.\n"
    "Mark 'match' as False if:\n"
    "- The post is discussing a tool, website, or app that has already been built, regardless of who created it (e.g., sharing a new app, asking for feedback on an app, promoting a product).\n"
    "- The post is not related to software, websites, or apps.\n"
    "- The post is a general discussion about a problem without explicitly seeking a software solution.\n"
    "- The post is asking for feedback on an idea the poster has already built.\n\n"
    "Answer in strict JSON with keys:\n"
    "{{\"match\": true|false, \"category\": \"saas|improve|pain|other\", \"reason\": \"<one-line reason>\"}}\n\n"
    "Post content:\n"
    "---\n"
    "{content}\n"
    "---\n"
    "Only output valid JSON."
)


SUMMARIZE_PROMPT = """You are an assistant. Summarize the following Reddit post in 2-3 sentences, focusing on the main point and any context that would help a reader understand the post.
Post content:
---
{content}
---
Only output the summary text.
"""


def fetch_feed(url: str, timeout: int = 10, max_retries: int = 5):
    headers = {
        "User-Agent": "reddit-post-summarizer/0.1 (by /u/yourusername)"
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            # If reddit returns Too Many Requests, wait 60s and retry
            if resp.status_code == 429:
                wait = 60
                print(
                    f"Received 429 for {url}. Waiting {wait} seconds before retry {attempt}/{max_retries}...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            xml = feedparser.parse(resp.content)
            return xml
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # For server errors, do an exponential backoff and retry
            if status and 500 <= status < 600 and attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"Server error {status} for {url}. Waiting {wait}s and retrying ({attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            # Otherwise re-raise the HTTP error
            raise
        except requests.exceptions.RequestException as e:
            # Network-level errors: retry with exponential backoff
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"Request failed ({e}). Waiting {wait}s and retrying ({attempt}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    # If we exhaust retries, raise a clear error
    raise requests.exceptions.HTTPError(
        f"Failed to fetch {url} after {max_retries} attempts")


def classify_post_text(text: str) -> Dict:
    prompt = CLASSIFY_PROMPT.format(content=text)
    resp = call_ollama(prompt)
    # try to parse JSON from response (best-effort)
    try:
        return json.loads(resp)
    except Exception:
        # naive extraction: look for braces
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(resp[start:end+1])
            except Exception:
                pass
        # fallback: no match
        return {"match": False, "category": "other", "reason": "could not parse LLM response"}


def summarize_post_text(text: str) -> str:
    prompt = SUMMARIZE_PROMPT.format(content=text)
    resp = call_ollama(prompt)
    return resp.strip()


def fetch_comments_and_analyze(candidate: Dict) -> Dict:
    """Fetches comments for a candidate post and analyzes them for solutions."""
    link = candidate["link"]
    comments_rss = link + ".rss"
    print("Checking comments for", candidate["title"], comments_rss)
    feed = fetch_feed(comments_rss)
    comments = []

    if feed and getattr(feed, "entries", None):
        for e in feed.entries:
            # skip the submission itself if present
            if getattr(e, "link", "") == candidate["link"]:
                continue

            comment_content = getattr(e, "content", "")
            if isinstance(comment_content, list):
                comment_content = comment_content[0].get("value", "")
            comments.append(comment_content)

    print(comments)
    if len(comments) == 0:
        print("No comments found for", candidate["title"])
        return {
            "title": candidate["title"],
            "link": candidate["link"],
            "content": candidate["content"],
            "classification": candidate["classification"],
            "comments_checked": len(comments),
        }

    analysis = comments_have_solution(comments)
    return {
        "title": candidate["title"],
        "link": candidate["link"],
        "content": candidate["content"],
        "classification": candidate["classification"],
        "comments_checked": len(comments),
        "comment_analysis": analysis,
    } if not analysis.get("found") else {
        "title": candidate["title"],
        "link": candidate["link"],
        "content": candidate["content"],
        "classification": candidate["classification"],
        "comments_checked": len(comments),
        "comment_analysis": analysis,
    }


def scan_subreddits_for_summaries(subreddits: List[str]) -> List[Dict]:
    """Fetches and summarizes posts from the specified subreddits."""
    posts = []
    for feed_url in subreddits:
        print("Fetching", feed_url)
        try:
            feed = fetch_feed(feed_url)
        except Exception as e:
            print(f"Failed to fetch {feed_url}: {e}")
            continue
        print("Feed fetched:", feed_url, "entries:",
              len(feed.entries) if feed else 0)
        if not feed:
            continue
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            content = getattr(entry, "content")
            full_text = f"Title: {title}\n\n{content}"
            summary = summarize_post_text(full_text)
            print("Summarized:", title)
            posts.append({
                "title": title,
                "full_text": full_text,
                "summary": summary,
                "link": link,
            })
    return posts


def comments_have_solution(comments: List[str]) -> Dict:
    joined = "\n\n".join(comments)[:20000]  # keep prompt size reasonable
    prompt = COMMENTS_PROMPT.format(comments=joined)
    resp = call_ollama(prompt)
    # try to parse JSON from response (best-effort)
    try:
        return json.loads(resp)
    except Exception:
        # naive extraction: look for braces
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(resp[start:end+1])
            except Exception:
                pass
        # fallback: no match
        return {"found": False, "example": "could not parse LLM response"}


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    clean_text = soup.get_text(separator=' ', strip=True)

    clean_text = re.sub(r'submitted by.*?comments\]', '', clean_text)
    clean_text = re.sub(r'&#32;', ' ', clean_text)
    clean_text = ''.join(char for char in clean_text if ord(char) < 128)

    return clean_text.strip()


def format_data(json_file_path: str) -> list:
    """
    Loads a JSON file, filters objects based on the 'match' key,
    and formats the data into a list of dictionaries with 'text' and 'target' keys.
    """
    formatted_list = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} contains invalid JSON.")
        return []

    for candidate in data:
        if 'classification' in candidate and 'match' in candidate['classification']:
            if candidate['classification']['match']:
                if 'content' in candidate and isinstance(candidate['content'], list) and len(candidate['content']) > 0:
                    content_value = clean_html(
                        candidate['content'][0].get('value', ''))
                    target = candidate['classification']['category']
                    formatted_list.append(
                        {"text": content_value, "target": "positive"})
            else:
                if 'content' in candidate and isinstance(candidate['content'], list) and len(candidate['content']) > 0:
                    content_value = clean_html(
                        candidate['content'][0].get('value', ''))
                    formatted_list.append(
                        {"text": content_value, "target": "negative"})
    return formatted_list


def classify_posts_from_json():
    input_path = "posts.json"
    output_path = "training-data.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    results = []
    for post in posts:
        print("Processing post:", post.get("title", ""))
        content = post.get("content", "")
        classification = classify_post_text(content)
        match = classification.get("match", False)
        target = "positive" if match else "negative"
        results.append({
            "text": content,
            "target": target
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # Replace with the actual path to your JSON file
    # file_path = "posts.json"
    # formatted_data = format_data(file_path)

    # # Save to a JSONL file
    # output_file_path = "training-data.jsonl"
    # with open(output_file_path, 'w', encoding='utf-8') as outfile:
    #     for entry in formatted_data:
    #         json.dump(entry, outfile, ensure_ascii=False)
    #         outfile.write('\n')  # Add newline for JSONL format

    # print(f"Formatted data saved to {output_file_path}")

    classify_posts_from_json()
