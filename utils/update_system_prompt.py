import argparse
import json
import os
import sys
import tempfile
from typing import Tuple


def load_prompt(prompt_path: str) -> str:
    try:
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read().strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}") from exc


def update_system_messages(record: dict, prompt_text: str) -> Tuple[dict, int]:
    messages = record.get("messages")
    replacements = 0

    if not isinstance(messages, list):
        return record, replacements

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            message["content"] = prompt_text
            replacements += 1

    return record, replacements


def process_dataset(input_path: str, output_path: str, prompt_path: str) -> Tuple[int, int]:
    prompt_text = load_prompt(prompt_path)
    total_records = 0
    total_replacements = 0

    temp_dir = os.path.dirname(os.path.abspath(output_path))
    with open(input_path, "r", encoding="utf-8") as infile, tempfile.NamedTemporaryFile(
        "w", delete=False, dir=temp_dir, encoding="utf-8"
    ) as temp_file:
        try:
            for line in infile:
                stripped = line.strip()
                if not stripped:
                    continue

                record = json.loads(stripped)
                record, replacements = update_system_messages(record, prompt_text)

                temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_records += 1
                total_replacements += replacements
        except json.JSONDecodeError as exc:
            temp_file.close()
            os.unlink(temp_file.name)
            raise json.JSONDecodeError(
                f"Failed to parse JSON on line {total_records + 1}: {exc.msg}",
                exc.doc,
                exc.pos,
            ) from exc

    temp_file_path = temp_file.name
    os.replace(temp_file_path, output_path)

    return total_records, total_replacements


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace system prompts in a JSONL dataset with the contents of prompt.txt."
    )
    parser.add_argument(
        "--input",
        default="dataset_formatted.jsonl",
        help="Path to the input JSONL dataset (default: dataset_formatted.jsonl).",
    )
    parser.add_argument(
        "--prompt",
        default="prompt.txt",
        help="Path to the file containing the replacement system prompt (default: prompt.txt).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the output JSONL file. If omitted, the input file is modified in place."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else input_path
    prompt_path = os.path.abspath(args.prompt)

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    total_records, total_replacements = process_dataset(input_path, output_path, prompt_path)

    print(f"Updated {total_records} records.")
    print(f"Replaced {total_replacements} system message(s).")
    print(f"Wrote output to: {output_path}")


if __name__ == "__main__":
    main()
