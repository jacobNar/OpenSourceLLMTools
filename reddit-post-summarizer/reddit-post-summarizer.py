from typing import List, Dict
import feedparser
import requests
import json
import time
from langchain_ollama import ChatOllama
subreddits = ["https://www.reddit.com/r/Doesthisexist/.rss",
              "https://www.reddit.com/r/apps/.rss", "https://www.reddit.com/r/smallbusiness/.rss"]


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
- a user asking for a Software-as-a-Service (SaaS),
- asking for an improved software/tool compared to what they use,
- or describing a business pain point that could be solved by software.

if it can be solved by software, mark match as True. If it is not related to software or tools, mark match as False.

Answer in strict JSON with keys:
{{"match": true|false, "category": "saas|improve|pain|other", "reason": "<one-line reason>"}}

Post content:
---
{content}
---
Only output valid JSON.
"""

COMMENTS_PROMPT = """You are an assistant. The following are comments from a Reddit post. Decide whether any comment provides a clear, actionable software solution suggestion (one that could be implemented as a SaaS or tool).
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
    return call_ollama(prompt)


def fetch_feed(url: str, timeout: int = 10):
    resp = requests.get(url)
    resp.raise_for_status()

    xml = feedparser.parse(resp.content)
    print(xml)
    return xml


def main():
    results = []
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
    with open("reddit_saas_candidates.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    # # For each candidate, fetch comments by adding .rss to the post link and scanning comments
    # final_list = []
    # for c in candidates:
    #     link = c["link"].rstrip("/")
    #     comments_rss = link + ".rss"
    #     print("Checking comments for", c["title"], comments_rss)
    #     feed = fetch_feed(comments_rss)
    #     comments = []
    #     if feed and getattr(feed, "entries", None):
    #         for e in feed.entries:
    #             # skip the submission itself if present
    #             if getattr(e, "link", "") == c["link"]:
    #                 continue
    #             comments.append(extract_entry_text(e))
    #     analysis = comments_have_solution(comments)
    #     if not analysis.get("found"):
    #         final_list.append({
    #             "title": c["title"],
    #             "link": c["link"],
    #             "classification": c["classification"],
    #             "comments_checked": len(comments),
    #             "comment_analysis": analysis,
    #         })
    #     time.sleep(0.5)

    # out_path = "reddit_saas_candidates.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(final_list, f, indent=2, ensure_ascii=False)

    # print(f"Done. {len(final_list)} items saved to {out_path}")
    # # minimal output
    # for i, item in enumerate(final_list, 1):
    #     print(i, item["title"], item["link"])


if __name__ == "__main__":
    main()
