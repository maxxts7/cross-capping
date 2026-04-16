#!/usr/bin/env python3
"""
Simple analysis of CSV files to understand the axis direction issue.
"""

import csv

def analyze_benign_degradation():
    """Analyze benign prompts where cross-axis capping was applied."""
    csv_file = "p25/cross_cap_benign_reclassified.csv"

    print("=== Cross-Axis Capping on Benign Prompts ===")

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        total_prompts = 0
        capped_prompts = 0
        degraded_prompts = 0

        for row in reader:
            total_prompts += 1

            correction_applied = row.get('correction_applied', '').strip()
            llm_label = row.get('llm_label', '').strip()
            prompt_idx = row.get('prompt_idx', '').strip()

            if correction_applied == 'Yes':
                capped_prompts += 1
                print(f"Prompt {prompt_idx}: Capped -> {llm_label}")

                if llm_label in ['false_refusal', 'degraded']:
                    degraded_prompts += 1

    print(f"\nSUMMARY:")
    print(f"Total benign prompts: {total_prompts}")
    print(f"Capped benign prompts: {capped_prompts}")
    print(f"Degraded benign prompts: {degraded_prompts}")
    print(f"Degradation rate: {degraded_prompts/total_prompts*100:.1f}%")

    if capped_prompts > 0:
        print(f"Degradation rate among capped: {degraded_prompts/capped_prompts*100:.1f}%")

def compare_assistant_vs_cross():
    """Compare assistant-axis vs cross-axis capping degradation rates."""
    files = {
        'assistant': "p25/assistant_cap_benign_reclassified.csv",
        'cross': "p25/cross_cap_benign_reclassified.csv"
    }

    print("\n=== Comparing Assistant vs Cross-Axis Capping ===")

    for method, csv_file in files.items():
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                total = 0
                capped = 0
                degraded = 0

                for row in reader:
                    total += 1

                    correction_applied = row.get('correction_applied', '').strip()
                    llm_label = row.get('llm_label', '').strip()

                    if correction_applied == 'Yes':
                        capped += 1

                    if llm_label in ['false_refusal', 'degraded']:
                        degraded += 1

                print(f"{method.upper()} capping:")
                print(f"  Total: {total}, Capped: {capped}, Degraded: {degraded}")
                print(f"  Capping rate: {capped/total*100:.1f}%")
                print(f"  Degradation rate: {degraded/total*100:.1f}%")

        except FileNotFoundError:
            print(f"File not found: {csv_file}")

if __name__ == "__main__":
    analyze_benign_degradation()
    compare_assistant_vs_cross()