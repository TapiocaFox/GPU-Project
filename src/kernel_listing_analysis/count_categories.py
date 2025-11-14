#!/usr/bin/env python3
import json

INPUT_FILE = "categorized.json"

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    keys = list(data.keys())
    print(f"total: {len(keys)}")   # number of top-level categories

    # For each category: print "key: count"
    for key in keys:
        items = data.get(key, [])
        print(f"{key}: {len(items)}")

if __name__ == "__main__":
    main()