import csv
import os
import json
import time
import re
from openai import OpenAI

INPUT_FILE = "kernel_list.csv"
OUTPUT_FILE = "categorized.json"
KEY_FILE = "open_ai_key.secret"
RAW_DIR = "raw_batches"
BATCH_SIZE = 300
RETRY_DELAY = 10  # seconds between retries on API error


def load_api_key():
    with open(KEY_FILE, "r") as f:
        return f.read().strip()


def safe_get(row, idx, default=""):
    """Safely get CSV column."""
    return row[idx].strip() if len(row) > idx else default


def extract_json_block(text):
    """Extract first {...} block from text."""
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def categorize_with_llm(client, kernels, batch_index=0):
    """Use LLM to categorize a batch of kernel names with full info."""
    os.makedirs(RAW_DIR, exist_ok=True)

    # Keep full row info
    kernel_text = "\n".join([f"- {row}" for row in [", ".join(k) for k in kernels]])

    prompt = f"""
You are categorizing CUDA GPU kernels based on their purpose.
Here is a list of kernel rows from CSV (all columns preserved):

{kernel_text}

Group them into logical categories (for example: Distance Kernels,
Binary Kernels, Dot Kernels, Utility Kernels, etc.)

Output strictly valid JSON in this format:
{{
  "CategoryName": ["full CSV row 1", "full CSV row 2", ...],
  "CategoryName2": [...]
}}
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in GPU kernel classification."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )
            break
        except Exception as e:
            print(f"❌ API error on batch {batch_index} (attempt {attempt+1}): {e}")
            if attempt < 2:
                print(f"⏳ Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return {}

    text = response.choices[0].message.content.strip()
    raw_path = os.path.join(RAW_DIR, f"batch_{batch_index}.txt")
    with open(raw_path, "w") as f:
        f.write(text)

    text = extract_json_block(text)

    try:
        parsed = json.loads(text)
        print(f"✅ Batch {batch_index} parsed successfully ({len(parsed)} categories).")
        return parsed
    except json.JSONDecodeError as e:
        print(f"⚠️ Invalid JSON in batch {batch_index} ({e}), saved raw text to {raw_path}.")
        return None


def main():
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # Load kernels: keep full row for each kernel
    kernels = []
    with open(INPUT_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 3:
                kernels.append(row)

    print(f"📦 Loaded {len(kernels)} kernels.")

    # Load checkpoint if exists
    all_results = {}
    processed_rows = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_results = json.load(f)
            # Keep track of already processed rows
            for rows in all_results.values():
                processed_rows.update(rows)
        print(f"🔄 Resuming from existing checkpoint with {len(processed_rows)} kernels already categorized.")

    for i in range(0, len(kernels), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE + 1
        batch = [k for k in kernels[i:i + BATCH_SIZE] if ", ".join(k) not in processed_rows]

        if not batch:
            print(f"➡️ Batch {batch_idx} already processed, skipping.")
            continue

        print(f"\n🧠 Processing batch {batch_idx} ({len(batch)} kernels)...")
        batch_result = categorize_with_llm(client, batch, batch_index=batch_idx)
        if not batch_result:
            continue

        # Merge valid categories
        for cat, rows in batch_result.items():
            all_results.setdefault(cat, []).extend(rows)
            processed_rows.update(rows)

        # Save intermediate checkpoint
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=4)

        print(f"💾 Saved checkpoint after batch {batch_idx}.")
        time.sleep(1.5)

    print(f"\n✅ Categorization complete. Final saved to {OUTPUT_FILE}.")
    print("\n=== Summary ===")
    for cat, rows in sorted(all_results.items()):
        print(f"{cat}: {len(rows)} kernels")


if __name__ == "__main__":
    main()