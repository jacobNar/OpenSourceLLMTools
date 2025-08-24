import json
from bs4 import BeautifulSoup
import re
from typing import List, Dict
from langchain_ollama import ChatOllama

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
    output_path = "training-data-summarize.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    results = []
    for post in posts:
        print("Processing post:", post.get("title", ""))
        summary = post.get("summary", "")
        classification = classify_post_text(summary)
        match = classification.get("match", False)
        target = "positive" if match else "negative"
        results.append({
            "text": summary,
            "target": target
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # Replace with the actual path to your JSON file
    # file_path = "reddit_saas_candidates.json"
    # formatted_data = format_data(file_path)

    # # Save to a JSONL file
    # output_file_path = "training-data.jsonl"
    # with open(output_file_path, 'w', encoding='utf-8') as outfile:
    #     for entry in formatted_data:
    #         json.dump(entry, outfile, ensure_ascii=False)
    #         outfile.write('\n')  # Add newline for JSONL format

    # print(f"Formatted data saved to {output_file_path}")

    classify_posts_from_json()
