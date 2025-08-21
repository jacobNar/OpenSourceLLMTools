import json
from bs4 import BeautifulSoup
import re


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    clean_text = soup.get_text(separator=' ', strip=True)

    clean_text = re.sub(r'submitted by.*?comments\]', '', clean_text)
    clean_text = re.sub(r'&#32;', ' ', clean_text)

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
                        {"text": content_value, "target": target})
            else:
                if 'content' in candidate and isinstance(candidate['content'], list) and len(candidate['content']) > 0:
                    content_value = clean_html(
                        candidate['content'][0].get('value', ''))
                    formatted_list.append(
                        {"text": content_value, "target": "negative"})
    return formatted_list


if __name__ == '__main__':
    # Replace with the actual path to your JSON file
    file_path = "reddit_saas_candidates.json"
    formatted_data = format_data(file_path)

    # Save to a JSONL file
    output_file_path = "training-data.jsonl"
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for entry in formatted_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')  # Add newline for JSONL format

    print(f"Formatted data saved to {output_file_path}")
