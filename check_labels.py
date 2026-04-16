#!/usr/bin/env python3
"""
Check what the actual LLM labels are in the CSV files.
"""

import csv

def check_unique_labels():
    """Check all unique LLM labels in both benign files."""
    files = {
        'assistant': "p25/assistant_cap_benign_reclassified.csv",
        'cross': "p25/cross_cap_benign_reclassified.csv"
    }

    for method, csv_file in files.items():
        print(f"\n=== {method.upper()} CAPPING LABELS ===")

        labels = set()
        correction_statuses = set()

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    llm_label = row.get('llm_label', '').strip()
                    correction_applied = row.get('correction_applied', '').strip()

                    labels.add(llm_label)
                    correction_statuses.add(correction_applied)

                    # Print a few examples
                    prompt_idx = row.get('prompt_idx', '').strip()
                    if prompt_idx in ['0', '1', '2']:
                        print(f"Prompt {prompt_idx}: correction={correction_applied}, label={llm_label}")

            print(f"All correction_applied values: {sorted(correction_statuses)}")
            print(f"All llm_label values: {sorted(labels)}")

        except FileNotFoundError:
            print(f"File not found: {csv_file}")

if __name__ == "__main__":
    check_unique_labels()