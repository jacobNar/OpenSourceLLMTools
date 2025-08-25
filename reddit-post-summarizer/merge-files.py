import json
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


SUMMARIZE_PROMPT = """You are an assistant. Summarize the following Reddit post in 2-3 sentences, focusing on the main point and any context that would help a reader understand the post.
Post content:
---
{content}
---
Only output the summary text.
"""


def summarize_post_text(text: str) -> str:
    prompt = SUMMARIZE_PROMPT.format(content=text)
    resp = call_ollama(prompt)
    return resp.strip()


def merge_positive_rows(input_file, output_file):
    """
    Opens an input JSONL file, iterates through each row,
    and appends rows with 'target' = 'positive' to an output JSONL file.
    """
    positive_rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get('target') == 'positive':
                    positive_rows.append(row)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    with open(output_file, 'a', encoding='utf-8') as f:
        for row in positive_rows:
            summary = summarize_post_text(row['text'])
            # print("Summary:", summary)
            row['text'] = summary
            json.dump(row, f)
            f.write('\n')


# Example usage:
merge_positive_rows('training-data.jsonl', 'training-data-summarize.jsonl')
