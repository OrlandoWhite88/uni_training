from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from threading import Lock, Semaphore
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

@dataclass
class ProcessingConfiguration:
    openai_client: OpenAI
    model_name: str
    concurrency_semaphore: Semaphore
    maximum_retries: int
    initial_retry_delay: float
    maximum_retry_delay: float
    reasoning_effort_level: str
    reasoning_summary_option: str
    delay_between_requests: float
    output_lock: Lock


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def build_reasoning_prompt(
    system_context: str,
    user_message: str,
    assistant_response: str,
) -> str:
    """
    Construct the audit prompt for the reasoning model.

    The model must keep the option selection fixed, but produce:
      - a detailed internal chain-of-thought ("thinking")
      - an outward-facing summary aligned with the selected confidence band
      - a numeric confidence value consistent with the summary
    """
    context_block = system_context.strip() or "Not provided."
    
    # Parse assistant response to detect task type
    try:
        parsed_response = json.loads(assistant_response)
        has_chapters = "chapters" in parsed_response and isinstance(parsed_response.get("chapters"), list)
    except:
        has_chapters = False

    if has_chapters:
        return f"""
You are auditing an HS code chapter selection decision.

SYSTEM / DEVELOPER CONTEXT:
{context_block}

USER MESSAGE (JSON):
{user_message}

EXISTING ASSISTANT OUTPUT (JSON):
{assistant_response}

Your tasks:
1. Analyse the chapter selection using the global confidence framework in the system context.
2. Keep the selected chapters exactly as-is. You must NOT alter which chapters are selected.
3. For EACH chapter in the list, determine the correct confidence score (0.00–1.00) with high precision.
4. Update the reasoning for each chapter to match its confidence band.
5. Produce a JSON object with EXACTLY these fields:
{{
  "thinking": "... detailed internal reasoning for chapter selection ...",
  "summary": "... overall explanation of chapter selection strategy ...",
  "chapters": [
    {{"chapter": "XX", "confidence": 0.XX, "reasoning": "brief explanation"}},
    {{"chapter": "YY", "confidence": 0.YY, "reasoning": "brief explanation"}}
  ]
}}

Constraints:
- Maintain the same chapter numbers in the same order.
- Each chapter's reasoning should align with its confidence level.
- The thinking field should explain the overall chapter selection strategy.
- Return ONLY the specified JSON object (no prose around it).
    """.strip()
    
    return f"""
You are auditing an HS code classification decision.

SYSTEM / DEVELOPER CONTEXT:
{context_block}

USER MESSAGE (JSON):
{user_message}

EXISTING ASSISTANT OUTPUT (JSON):
{assistant_response}

Your tasks:
1. Analyse the decision using the global confidence framework contained in the system context.
2. Keep the selected option exactly as-is. You must NOT alter the option number or add/remove fields other than the requested updates.
3. Determine the correct confidence score (0.00–1.00) with high precision based only on the CURRENT decision level.
4. Write a short external-facing summary that matches the chosen confidence band (e.g., if confidence ≥ 0.85 the summary must not claim it is below threshold).
5. Produce a JSON object with EXACTLY these fields:
{{
  "thinking": "... detailed internal reasoning ...",
  "summary": "... outward-facing explanation consistent with confidence ...",
  "confidence": 0.00
}}

Constraints:
- The internal thinking can reference the confidence framework, but the summary should be concise, user-facing, and free of contradictions.
- If the confidence is ≥ 0.85, describe the decision as proceed-ready / confident.
- If the confidence is < 0.85, explain which missing details prevent proceeding.
- Do not mention these instructions or the existence of this audit.
- Return ONLY the specified JSON object (no prose around it).
    """.strip()


def extract_first_output_text(response: Any) -> Optional[str]:
    """Extract the first text block from the Responses API result."""
    output = getattr(response, "output", None)
    if not output:
        return getattr(response, "output_text", None)

    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            if isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        return text
    return None


def is_retryable_exception(exc: Exception) -> bool:
    """Determine if an exception is retryable."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError)):
        return True

    message = str(exc).lower()
    retry_keywords = (
        "connection",
        "timeout",
        "temporarily",
        "try again",
        "reset",
        "unavailable",
        "503",
        "502",
        "504",
        "429",
        "rate limit",
        "timed out",
        "backend fetch failed",
    )
    return any(keyword in message for keyword in retry_keywords)


def generate_reasoning(
    system_context: str,
    user_message: str,
    assistant_response: str,
    config: ProcessingConfig,
) -> Optional[Dict[str, Any]]:
    """Call the Responses API and return the reasoning payload as a dict."""
    prompt = build_reasoning_prompt(system_context, user_message, assistant_response)

    attempt = 1
    delay = config.initial_backoff

    while attempt <= config.max_retries:
        try:
            with config.semaphore:
                response = config.client.responses.create(
                    model=config.model,
                    input=prompt,
                    text={
                        "format": {"type": "json_object"},
                        "verbosity": "medium",
                    },
                    reasoning={
                        "effort": config.reasoning_effort,
                        "summary": config.reasoning_summary,
                    },
                    store=False,
                )

            payload = extract_first_output_text(response)
            if not payload:
                raise ValueError("Empty response content from reasoning model.")

            result = json.loads(payload)

            if config.sleep_between_requests > 0:
                time.sleep(config.sleep_between_requests)

            return result

        except json.JSONDecodeError as exc:
            with config.print_lock:
                print(f"[reasoning] JSON decoding failed: {exc}", file=sys.stderr, flush=True)
            return None
        except Exception as exc:
            retryable = is_retryable_exception(exc)
            if not retryable or attempt == config.max_retries:
                with config.print_lock:
                    print(f"[reasoning] Attempt {attempt} failed and will NOT retry: {exc}", file=sys.stderr, flush=True)
                return None

            sleep_for = min(delay, config.max_backoff)
            with config.print_lock:
                print(
                    f"[reasoning] Attempt {attempt} failed ({exc}); retrying in {sleep_for:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(sleep_for)
            delay = min(delay * 2, config.max_backoff)
            attempt += 1

    return None


def process_single_example(line_num: int, line: str, config: ProcessingConfig) -> Dict[str, Any]:
    """Process a single JSONL entry."""
    try:
        example = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"line_num": line_num, "success": False, "error": f"JSON decode error: {exc}"}

    messages: List[dict] = example.get("messages", [])
    if not messages:
        return {"line_num": line_num, "success": False, "error": "Missing 'messages' array."}

    system_context_parts: List[str] = []
    user_content: str = ""
    assistant_msg: Optional[Dict[str, Any]] = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system" or role == "developer":
            system_context_parts.append(normalise_message_content(content))
        elif role == "user":
            user_content = normalise_message_content(content)
        elif role == "assistant":
            assistant_msg = msg

    if assistant_msg is None:
        return {"line_num": line_num, "success": False, "error": "Missing assistant message."}

    assistant_content_raw = assistant_msg.get("content", "")
    assistant_content_str = normalise_message_content(assistant_content_raw)

    system_context = "\n\n".join(part for part in system_context_parts if part).strip()

    reasoning_payload = generate_reasoning(system_context, user_content, assistant_content_str, config)
    if reasoning_payload is None:
        return {"line_num": line_num, "success": False, "error": "Failed to generate reasoning."}

    thinking_text = reasoning_payload.get("thinking", "").strip()
    summary_text = reasoning_payload.get("summary", "").strip() or thinking_text
    confidence_value = reasoning_payload.get("confidence")
    chapters_list = reasoning_payload.get("chapters")
    
    try:
        confidence_value = float(confidence_value) if confidence_value is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    if thinking_text:
        assistant_msg["thinking"] = thinking_text

    try:
        assistant_content = assistant_msg.get("content")
        if isinstance(assistant_content, str):
            parsed_content = json.loads(assistant_content)
            
            # Handle select_chapters format
            if chapters_list is not None and isinstance(chapters_list, list):
                parsed_content["chapters"] = chapters_list
                if summary_text:
                    parsed_content["reasoning"] = summary_text
            else:
                # Handle other task formats
                if confidence_value is not None:
                    parsed_content["confidence"] = confidence_value
                if summary_text:
                    parsed_content["reasoning"] = summary_text
            
            assistant_msg["content"] = json.dumps(parsed_content, ensure_ascii=False)
        elif isinstance(assistant_content, dict):
            # Handle select_chapters format
            if chapters_list is not None and isinstance(chapters_list, list):
                assistant_content["chapters"] = chapters_list
                if summary_text:
                    assistant_content["reasoning"] = summary_text
            else:
                # Handle other task formats
                if confidence_value is not None:
                    assistant_content["confidence"] = confidence_value
                if summary_text:
                    assistant_content["reasoning"] = summary_text
            
            assistant_msg["content"] = assistant_content
    except json.JSONDecodeError as exc:
        with config.print_lock:
            print(f"[line {line_num}] Warning: assistant content is not valid JSON ({exc})", file=sys.stderr, flush=True)

    return {"line_num": line_num, "success": True, "data": example}


def process_dataset(
    input_file: str,
    output_file: str,
    config: ProcessingConfig,
    max_examples: Optional[int],
    max_workers: int,
    failed_output: Optional[str],
) -> None:
    """Process the entire dataset with optional parallelism."""
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    if max_examples is not None:
        lines = lines[:max_examples]

    total = len(lines)
    if total == 0:
        print("No examples found in the input file.")
        return

    max_workers = max(1, min(max_workers, total))

    failures: List[Dict[str, Any]] = []
    success_count = 0
    outfile_lock = Lock()

    with open(output_file, "w", encoding="utf-8") as outfile:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single_example, idx, line, config): idx for idx, line in enumerate(lines)
            }

            with tqdm(total=total, desc="Processing dataset", unit="example", leave=True) as progress_bar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:  # noqa: BLE001
                        failures.append({"line_num": idx, "error": f"Unhandled exception: {exc}"})
                        with config.print_lock:
                            print(f"[line {idx}] Unhandled exception: {exc}", file=sys.stderr, flush=True)
                    else:
                        if outcome.get("success"):
                            with outfile_lock:
                                outfile.write(json.dumps(outcome["data"], ensure_ascii=False) + "\n")
                                outfile.flush()
                            success_count += 1
                        else:
                            failures.append({"line_num": outcome["line_num"], "error": outcome.get("error")})
                    finally:
                        progress_bar.update(1)

    print("\nProcessing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failures)}")
    print(f"Output written to: {output_file}")

    if failures:
        if failed_output:
            with open(failed_output, "w", encoding="utf-8") as failure_file:
                for failure in failures:
                    failure_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"Failure details written to: {failed_output}")
        else:
            print("Failure details (first 10 shown):")
            for failure in failures[:10]:
                print(f"  line {failure['line_num']}: {failure['error']}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more. Use --failed-output to store all failures.")


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a dataset with reasoning traces using the OpenAI Responses API."
    )
    parser.add_argument("--input", default="dataset_formatted.jsonl", help="Input JSONL file.")
    parser.add_argument("--output", default="dataset_with_reasoning.jsonl", help="Output JSONL file.")
    parser.add_argument("--model", default="gpt-5", help="Model ID to use (default: gpt-5).")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of examples to process.")
    parser.add_argument("--workers", type=int, default=100, help="Thread pool size (default: 20).")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=100,
        help="Maximum number of concurrent API calls (default: 5).",
    )
    parser.add_argument("--retries", type=int, default=5, help="Maximum number of retries per example.")
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=2.0,
        help="Initial backoff delay in seconds for retryable errors (default: 2.0).",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=30.0,
        help="Maximum backoff delay in seconds when retrying (default: 30.0).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep in seconds after each successful request (default: 0.0).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=("low", "medium", "high"),
        help="Reasoning effort level for the Responses API (default: medium).",
    )
    parser.add_argument(
        "--reasoning-summary",
        default="auto",
        choices=("auto", "none"),
        help="Controls whether the API should synthesize an outward summary (default: auto).",
    )
    parser.add_argument(
        "--failed-output",
        default=None,
        help="Optional JSONL path to save failures for later inspection.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only three examples (overrides --max if it is larger than 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_arguments()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    semaphore = Semaphore(max(1, args.max_concurrency))
    print_lock = Lock()

    config = ProcessingConfig(
        client=client,
        model=args.model,
        semaphore=semaphore,
        max_retries=args.retries,
        initial_backoff=args.initial_backoff,
        max_backoff=args.max_backoff,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
        sleep_between_requests=args.sleep,
        print_lock=print_lock,
    )

    max_examples = args.max
    if args.test:
        max_examples = 3 if max_examples is None else min(max_examples, 3)

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Max workers: {args.workers}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"Max examples: {max_examples if max_examples is not None else 'All'}")
    print()

    process_dataset(
        input_file=args.input,
        output_file=args.output,
        config=config,
        max_examples=max_examples,
        max_workers=args.workers,
        failed_output=args.failed_output,
    )


if __name__ == "__main__":
    main()
