from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Dict
import json

tokenizer = AutoTokenizer.from_pretrained("jacobNar/autotrain-vtwei-2s4ra")
model = AutoModelForSequenceClassification.from_pretrained(
    "jacobNar/autotrain-vtwei-2s4ra")
pipe = pipeline("text-classification",
                model="jacobNar/autotrain-vtwei-2s4ra", device="cuda")


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
        if (classification['label'] == "positive" and classification['score'] > 0.85):
            if post["link"] not in existing_links:
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
