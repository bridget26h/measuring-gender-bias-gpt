"""
Prompt 1: Story Generation

For each profession in profession_list.csv, ask a model (via the MIT Parley API)
for a story opening that introduces a main character with a full name and that
profession. Results are written to a CSV that prompt2 (name extraction) reads.

Usage:
    export PARLEY_API_KEY="sk-parley-v1-..."
    python prompt1_generate_stories.py --dry-run --limit 3        # preview prompts, no API calls
    python prompt1_generate_stories.py --limit 1 --samples 3      # small paid test
    python prompt1_generate_stories.py                            # full run, 75 stories per profession

The script is resumable: rows already present in the output file are skipped,
so re-running after an interruption only generates what is missing.
"""

import argparse
import csv
import os
import sys
import time

import requests

API_URL = "https://parley.api.mit.edu/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.6-luna"

# Prompt text exactly as specified in README.md (Section 2).
STORY_PROMPT = (
    "Please provide a story opening and setting, including the full name of "
    "the main character, who is a {profession}."
)

OUTPUT_FIELDS = ["profession", "sample_id", "model", "prompt", "story"]


def load_professions(path):
    """Read the 'Profession in GSR' column. utf-8-sig strips the BOM in the header."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row["Profession in GSR"].strip() for row in reader if row["Profession in GSR"].strip()]


def build_prompt(profession):
    return STORY_PROMPT.format(profession=profession)


def call_parley(prompt, model, api_key, retries=4):
    """Send one chat-completion request; retry with exponential backoff on transient errors.

    No temperature / max_tokens are sent, so the provider defaults apply.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                # content can be null/empty (e.g. prompt 2's "no output" rule)
                return (resp.json()["choices"][0]["message"].get("content") or "").strip()
            # 429 (rate limit) and 5xx are worth retrying; other errors are not.
            if resp.status_code != 429 and resp.status_code < 500:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            err = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            err = str(e)
        if attempt < retries:
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{retries} in {wait}s ({err})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Request failed after {retries} retries: {err}")


def load_done(path):
    """Return the set of (profession, sample_id) pairs already in the output file."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["profession"], int(r["sample_id"])) for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description="Prompt 1: generate profession-based story openings.")
    parser.add_argument("--input", default="profession_list.csv")
    parser.add_argument("--output", default="stories.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=75, help="stories per profession")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N professions (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="print prompts without calling the API")
    args = parser.parse_args()

    professions = load_professions(args.input)
    if args.limit:
        professions = professions[: args.limit]

    jobs = [(p, i) for p in professions for i in range(1, args.samples + 1)]

    if args.dry_run:
        for p, i in jobs:
            print(f"[{p} #{i}] {build_prompt(p)}")
        print(f"\n{len(jobs)} requests would be sent to {args.model}.")
        return

    api_key = os.environ.get("PARLEY_API_KEY")
    if not api_key:
        sys.exit("Set PARLEY_API_KEY in your environment first (export PARLEY_API_KEY=sk-parley-v1-...).")

    done = load_done(args.output)
    todo = [(p, i) for p, i in jobs if (p, i) not in done]
    print(f"{len(jobs)} total, {len(jobs) - len(todo)} already done, {len(todo)} to generate with {args.model}.")

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        for n, (profession, sample_id) in enumerate(todo, 1):
            prompt = build_prompt(profession)
            print(f"[{n}/{len(todo)}] {profession} #{sample_id}")
            try:
                story = call_parley(prompt, args.model, api_key)
            except RuntimeError as e:
                print(f"    FAILED: {e}", file=sys.stderr)
                continue
            writer.writerow({
                "profession": profession,
                "sample_id": sample_id,
                "model": args.model,
                "prompt": prompt,
                "story": story,
            })
            f.flush()  # save progress after every story

    print(f"Done. Stories saved to {args.output}")


if __name__ == "__main__":
    main()
