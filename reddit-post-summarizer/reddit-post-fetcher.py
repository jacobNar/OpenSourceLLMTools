from typing import List, Dict
from bs4 import BeautifulSoup
import feedparser
import requests
import json
import time
import re

subreddits = ["https://www.reddit.com/r/Doesthisexist/.rss",
              "https://www.reddit.com/r/apps/.rss", "https://www.reddit.com/r/smallbusiness/.rss",
              "https://www.reddit.com/r/ProductManagement/.rss", "https://www.reddit.com/r/SideProject/.rss", "https://www.reddit.com/r/SaaS/.rss",
              "https://www.reddit.com/r/Solopreneur/.rss"]


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
            if 'content' in entry and isinstance(entry['content'], list) and len(entry['content']) > 0:
                content = getattr(entry, "content")[0].get("value", "")
            else:
                content = ""

            print("Fetched:", title, link)
            candidates.append({
                "title": title,
                "link": link,
                "content": clean_html(content),
            })

    return candidates


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    clean_text = soup.get_text(separator=' ', strip=True)

    clean_text = re.sub(r'submitted by.*?comments\]', '', clean_text)
    clean_text = re.sub(r'&#32;', ' ', clean_text)
    clean_text = ''.join(char for char in clean_text if ord(char) < 128)

    return clean_text.strip()


def main():
    out_path = "posts.json"
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            final_list = json.load(f)
    except FileNotFoundError:
        final_list = []

    existing_links = set(item["link"] for item in final_list if "link" in item)
    # Only change: use new summarization flow
    posts = scan_subreddits(subreddits)

    added = 0
    for post in posts:
        if post["link"] not in existing_links:
            final_list.append(post)
            existing_links.add(post["link"])
            added += 1
        else:
            print(f"Skipping existing post: {post['title']} {post['link']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(final_list)} items saved to {out_path}")
    for i, item in enumerate(final_list, 1):
        print(i, item["title"], item["link"])


if __name__ == "__main__":
    main()
