import csv
import json
from openai import OpenAI

INPUT_FILE = "kernel_list.csv"
OUTPUT_FILE = "categorized.json"
KEY_FILE = "open_ai_key.secret"

def load_api_key():
    """Read API key from open_ai_key.secret (strip newlines/spaces)."""
    try:
        with open(KEY_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"API key file '{KEY_FILE}' not found.")

def categorize_with_llm(client, kernels):
    """
    Use an LLM to categorize kernel names and signatures.
    """
    kernel_text = "\n".join([f"- {k[2]}: {k[4]}" for k in kernels])

    prompt = f"""
You are categorizing CUDA GPU kernels based on their purpose.
Here is a list of kernel function names and their arguments:

{kernel_text}

Group them into logical categories (for example: Distance Kernels,
Binary Kernels, Dot Kernels, Utility Kernels, etc.)

Output strictly valid JSON in this format:
{{
  "CategoryName": ["kernel1", "kernel2", ...],
  "CategoryName2": [...]
}}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in GPU kernel classification."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ Warning: LLM response was not valid JSON. Saving raw text.")
        return {"Unparsed": [text]}

def main():
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    kernels = []
    with open(INPUT_FILE, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or len(row) < 5:
                continue
            kernels.append(row)

    categorized = categorize_with_llm(client, kernels)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(categorized, f, indent=4)

    print("\n=== LLM Categorization Results ===")
    for cat, names in categorized.items():
        print(f"\n[{cat}]")
        for n in names:
            print(f"  - {n}")

if __name__ == "__main__":
    main()