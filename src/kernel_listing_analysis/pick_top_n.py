#!/usr/bin/env python3
import json
import math

INPUT_FILE = "categorized.json"

TOP_N_CATEGORY = 20
PICK_N_TEST_ITEMS = 100
PICK_N_VALID_ITEMS = 100
PICK_N_TOTAL = PICK_N_TEST_ITEMS + PICK_N_VALID_ITEMS


OUTPUT_TRAIN = "train.json"
OUTPUT_VALID = "validation.json"
OUTPUT_TEST = "test.json"


def proportional_pick(counts, total):
    """Return proportional allocations for each category."""
    sum_counts = sum(counts)
    raw = [(c / sum_counts) * total for c in counts]

    picks = [max(1, math.floor(x)) for x in raw]
    diff = total - sum(picks)

    # Add items where needed
    while diff > 0:
        for i in sorted(range(len(counts)), key=lambda j: -counts[j]):
            picks[i] += 1
            diff -= 1
            if diff == 0:
                break

    # Remove if too many
    while diff < 0:
        for i in sorted(range(len(counts)), key=lambda j: counts[j]):
            if picks[i] > 1:
                picks[i] -= 1
                diff += 1
                if diff == 0:
                    break

    return picks


def split_items(items, n_test, n_valid):
    """Split items disjointly: first test, then valid, rest train."""
    test_items = items[:n_test]
    valid_items = items[n_test:n_test + n_valid]
    train_items = items[n_test + n_valid:]
    return test_items, valid_items, train_items


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Sort categories by size
    sorted_items = sorted(data.items(), key=lambda kv: -len(kv[1]))

    # Top N categories
    top = sorted_items[:TOP_N_CATEGORY]
    cat_names = [name for name, _ in top]
    counts = [len(items) for _, items in top]

    # Compute proportional allocations
    test_picks = proportional_pick(counts, PICK_N_TEST_ITEMS)
    valid_picks = proportional_pick(counts, PICK_N_VALID_ITEMS)

    # Build outputs
    test_result = {}
    valid_result = {}
    train_result = {}

    for (cat, items), n_test, n_valid in zip(top, test_picks, valid_picks):
        test_items, valid_items, train_items = split_items(items, n_test, n_valid)

        test_result[cat] = test_items
        valid_result[cat] = valid_items
        train_result[cat] = train_items

    # Write to files
    with open(OUTPUT_TEST, "w") as f:
        json.dump(test_result, f, indent=4)

    with open(OUTPUT_VALID, "w") as f:
        json.dump(valid_result, f, indent=4)

    with open(OUTPUT_TRAIN, "w") as f:
        json.dump(train_result, f, indent=4)

    print("Wrote", OUTPUT_TEST)
    print("Wrote", OUTPUT_VALID)
    print("Wrote", OUTPUT_TRAIN)


if __name__ == "__main__":
    main()