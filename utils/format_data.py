from typing import Any, Dict, List
import json
import re

def load_jsonl_examples(example_file_path: str) -> List[Dict[str, Any]]:
    loaded_examples = []
    with open(example_file_path, encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped_line = line.strip()
            if stripped_line:
                parsed_example = json.loads(stripped_line)
                if isinstance(parsed_example, dict):
                    loaded_examples.append(parsed_example)
    return loaded_examples

def save_jsonl_examples(examples: List[Dict[str, Any]], output_file_path: str) -> None:
    with open(output_file_path, "w", encoding="utf-8") as file_handle:
        for example in examples:
            json_string = json.dumps(example, ensure_ascii=False, separators=(",", ":"))
            file_handle.write(json_string + "\n")

def extract_clean_json(raw_response: str) -> str:
    markdown_json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    pattern_match = re.search(markdown_json_pattern, raw_response)
    if pattern_match:
        extracted_json = pattern_match.group(1).strip()
    else:
        extracted_json = raw_response.strip()
    parsed_data = json.loads(extracted_json)
    return json.dumps(parsed_data, separators=(",", ":"))

def format_dataset_for_gpt_oss(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted_examples = []
    for example in examples:
        original_messages = example.get("messages", [])
        processed_messages = []
        for message in original_messages:
            message_role = message.get("role")
            message_content = message.get("content", "")
            if message_role == "system":
                processed_messages.append({
                    "role": "system",
                    "content": message_content,
                    "thinking": None
                })
            elif message_role == "user":
                processed_messages.append({
                    "role": "user",
                    "content": message_content,
                    "thinking": None
                })
            elif message_role == "assistant":
                cleaned_content = extract_clean_json(message_content)
                processed_messages.append({
                    "role": "assistant",
                    "content": cleaned_content,
                    "thinking": None
                })
        if processed_messages:
            formatted_examples.append({"messages": processed_messages})
    return formatted_examples

def main() -> None:
    source_file = "dataset.jsonl"
    destination_file = "dataset_formatted.jsonl"
    source_examples = load_jsonl_examples(source_file)
    transformed_examples = format_dataset_for_gpt_oss(source_examples)
    save_jsonl_examples(transformed_examples, destination_file)

if __name__ == "__main__":
    main()
