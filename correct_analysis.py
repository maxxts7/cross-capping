#!/usr/bin/env python3
"""
Correct analysis using the right label names.
"""

import csv

def analyze_degradation_correctly():
    """Analyze degradation using correct label names."""
    files = {
        'assistant': "p25/assistant_cap_benign_reclassified.csv",
        'cross': "p25/cross_cap_benign_reclassified.csv"
    }

    print("=== CORRECT DEGRADATION ANALYSIS ===")

    for method, csv_file in files.items():
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                total = 0
                degraded = 0

                for row in reader:
                    total += 1
                    llm_label = row.get('llm_label', '').strip()

                    if llm_label == 'benign_degraded':
                        degraded += 1

                print(f"{method.upper()} capping:")
                print(f"  Total: {total}")
                print(f"  Degraded: {degraded}")
                print(f"  Degradation rate: {degraded/total*100:.1f}%")
                print()

        except FileNotFoundError:
            print(f"File not found: {csv_file}")

if __name__ == "__main__":
    analyze_degradation_correctly()