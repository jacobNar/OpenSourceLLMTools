from typing import List, Dict
import feedparser
import requests
import json
import time
from langchain_ollama import ChatOllama
subreddits = ["https://www.reddit.com/r/Doesthisexist/.rss",
              "https://www.reddit.com/r/apps/.rss", "https://www.reddit.com/r/smallbusiness/.rss",
              "https://www.reddit.com/r/ProductManagement/.rss", "https://www.reddit.com/r/SideProject/.rss", "https://www.reddit.com/r/SaaS/.rss"
              "https://www.reddit.com/r/Solopreneur/.rss"]


OLLAMA_MODEL = "llama3.2"


def call_ollama(prompt: str, max_tokens: int = 2048) -> str:
    try:
        # instantiate ChatOllama client and call .invoke with messages list
        llm = ChatOllama(
            base_url="http://localhost:11434/",
            model=OLLAMA_MODEL,
            temperature=0.2,
            num_predict=max_tokens,
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


CLASSIFY_PROMPT = """You are an assistant. Given the following Reddit post content, decide whether the post is:
- a user asking for a recommendation for software, a website, or an app.
- asking for an improved software/tool compared to what they use.
- or describing a business pain point that could be solved by software.

If the post is a request for a software, website, or app solution, whether it's for a new recommendation or an improvement on an existing one, mark "match" as True.

Mark "match" as False if:
- The post is discussing a tool, website, or app that has already been built, regardless of who created it (e.g., sharing a new app, asking for feedback on an app, promoting a product).
- The post is not related to software, websites, or apps.
- The post is a general discussion about a problem without explicitly seeking a software solution.

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


def scan_subreddits(subreddits: List[str]) -> List[Dict]:
    """Fetches and classifies posts from the specified subreddits."""
    candidates = []
    for feed_url in subreddits:
        print("Fetching", feed_url)
        feed = fetch_feed(feed_url)
        print("Feed fetched:", feed_url, "entries:",
              len(feed.entries) if feed else 0)
        if not feed:
            continue
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            content = getattr(entry, "content")
            full_text = f"Title: {title}\n\n{content}"
            classification = classify_post_text(full_text)
            print("Classified:", title, classification)
            if classification.get("match"):
                candidates.append({
                    "title": title,
                    "link": link,
                    "content": content,
                    "classification": classification,
                })
    return candidates


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


def main():
    out_path = "reddit_saas_candidates.json"
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            final_list = json.load(f)
    except FileNotFoundError:
        final_list = []

    existing_links = {item["link"] for item in final_list}

    candidates = scan_subreddits(subreddits)

    # loop through candidates, analyze them and update if exists or add new one
    for c in candidates:
        analyzed_post = fetch_comments_and_analyze(c)
        if analyzed_post["link"] in existing_links:
            # Update existing post
            for i, item in enumerate(final_list):
                if item["link"] == analyzed_post["link"]:
                    final_list[i] = analyzed_post
                    break
        else:
            # Append new post
            final_list.append(analyzed_post)
        time.sleep(0.5)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(final_list)} items saved to {out_path}")
    # minimal output
    for i, item in enumerate(final_list, 1):
        print(i, item["title"], item["link"])


if __name__ == "__main__":
    main()
