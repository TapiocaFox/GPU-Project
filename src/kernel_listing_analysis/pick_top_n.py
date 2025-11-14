#!/usr/bin/env python3
import json
import math

INPUT_FILE = "categorized.json"
OUTPUT_FILE = "top_n.json"

TOP_N_CATEGORY = 20
PICK_N_ITEMS = 100


def proportional_pick(category_counts, pick_total):
    """Return how many items to pick per category proportionally."""
    total_items = sum(category_counts)
    raw_alloc = [(c / total_items) * pick_total for c in category_counts]

    # First assign floor
    picks = [max(1, math.floor(x)) for x in raw_alloc]

    # Adjust if sum mismatch
    diff = pick_total - sum(picks)

    # If we need to add items
    while diff > 0:
        # give to largest categories first
        for i in sorted(range(len(category_counts)), key=lambda j: -category_counts[j]):
            picks[i] += 1
            diff -= 1
            if diff == 0:
                break

    # If we need to remove items
    while diff < 0:
        for i in sorted(range(len(category_counts)), key=lambda j: category_counts[j]):
            if picks[i] > 1:
                picks[i] -= 1
                diff += 1
                if diff == 0:
                    break

    return picks


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Sort categories by descending count
    sorted_items = sorted(data.items(), key=lambda kv: -len(kv[1]))

    # Pick the top N categories
    top = sorted_items[:TOP_N_CATEGORY]

    cats = [name for name, items in top]
    counts = [len(items) for _, items in top]

    # Compute how many items to pick from each
    picks = proportional_pick(counts, PICK_N_ITEMS)

    # Build resulting data
    result = {}
    for (cat, items), n in zip(top, picks):
        result[cat] = items[:n]  # take first n items

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=4)

    print("Wrote", OUTPUT_FILE)


if __name__ == "__main__":
    main()