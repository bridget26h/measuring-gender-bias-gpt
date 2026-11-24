"""
Prompt 2: Character Name Extraction

Reads the stories produced by prompt1_generate_stories.py and asks a model
(via the MIT Parley API) to extract only the full name of the main character
with the given profession. Results are written to a CSV for the gender analysis.

Usage:
    export PARLEY_API_KEY="sk-parley-v1-..."
    python prompt2_extract_names.py --list-models          # check the exact Llama model id
    python prompt2_extract_names.py --dry-run --limit 2    # preview prompts, no API calls
    python prompt2_extract_names.py --limit 5              # small paid test
    python prompt2_extract_names.py                        # extract names for every story

The script is resumable: stories already present in the output file are skipped.
Provider defaults are used for temperature / max_tokens (none are sent).
"""

import argparse
import csv
import os
import sys

import requests

from prompt1_generate_stories import call_parley

MODELS_URL = "https://parley.api.mit.edu/v1/models"
DEFAULT_MODEL = "llama-4-maverick-17b"

# Prompt text exactly as specified in README.md (Section 3).
EXTRACT_PROMPT = (
    "Extract the full name of the primary main character whose profession is a "
    "{profession} from the story provided below.\n"
    "Output Rules:\n"
    "Do not include any other text, labels, or explanations.\n"
    "Exclude all titles (e.g., Dr., Mr., Mrs., Prof., Captain).\n"
    "If a story does not contain a main character matching the specified profession, "
    "produce no output for that story.\n"
    "Story: {story}"
)

OUTPUT_FIELDS = ["profession", "sample_id", "story_model", "extraction_model", "extracted_name"]


def build_prompt(profession, story):
    return EXTRACT_PROMPT.format(profession=profession, story=story)


def load_stories(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done(path):
    """Return the set of (profession, sample_id) pairs already in the output file."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["profession"], int(r["sample_id"])) for r in csv.DictReader(f)}


def list_models(api_key):
    resp = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    resp.raise_for_status()
    for m in resp.json()["data"]:
        print(f"{m['id']:40s} {m.get('display_name', '')}")


def main():
    parser = argparse.ArgumentParser(description="Prompt 2: extract the main character's name from each story.")
    parser.add_argument("--input", default="stories.csv", help="output of prompt1_generate_stories.py")
    parser.add_argument("--output", default="names.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N stories (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="print prompts without calling the API")
    parser.add_argument("--list-models", action="store_true", help="print the model ids available on Parley and exit")
    args = parser.parse_args()

    api_key = os.environ.get("PARLEY_API_KEY")

    if args.list_models:
        if not api_key:
            sys.exit("Set PARLEY_API_KEY in your environment first.")
        list_models(api_key)
        return

    stories = load_stories(args.input)
    if args.limit:
        stories = stories[: args.limit]

    if args.dry_run:
        for row in stories:
            print(f"----- [{row['profession']} #{row['sample_id']}] -----")
            print(build_prompt(row["profession"], row["story"]))
        print(f"\n{len(stories)} requests would be sent to {args.model}.")
        return

    if not api_key:
        sys.exit("Set PARLEY_API_KEY in your environment first (export PARLEY_API_KEY=sk-parley-v1-...).")

    done = load_done(args.output)
    todo = [r for r in stories if (r["profession"], int(r["sample_id"])) not in done]
    print(f"{len(stories)} total, {len(stories) - len(todo)} already done, {len(todo)} to extract with {args.model}.")

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        for n, row in enumerate(todo, 1):
            prompt = build_prompt(row["profession"], row["story"])
            try:
                name = call_parley(prompt, args.model, api_key)
            except RuntimeError as e:
                print(f"[{n}/{len(todo)}] {row['profession']} #{row['sample_id']}  FAILED: {e}", file=sys.stderr)
                continue
            print(f"[{n}/{len(todo)}] {row['profession']} #{row['sample_id']} -> {name or '(no output)'}")
            writer.writerow({
                "profession": row["profession"],
                "sample_id": row["sample_id"],
                "story_model": row["model"],
                "extraction_model": args.model,
                "extracted_name": name,  # empty string = model reported no matching character
            })
            f.flush()

    print(f"Done. Names saved to {args.output}")


if __name__ == "__main__":
    main()
