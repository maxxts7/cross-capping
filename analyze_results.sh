#!/bin/bash

# This script analyzes the reclassified CSV files
# CSV structure: prompt_idx,prompt_text,baseline_text,correction_applied,layers,capped_text,llm_label,llm_judge_model

cd "Final Results"

for csvfile in *_reclassified.csv; do
    echo ""
    echo "======================================================================="
    echo "FILE: $csvfile"
    echo "======================================================================="
    
    total=$(($(wc -l < "$csvfile") - 1))
    echo "Total Rows: $total"
    echo ""
    
    # Count patterns at end of lines for labels
    echo "LABELS (searching for pattern at line endings):"
    grep -o '\(refusal\|compliance\|benign_unchanged\|benign_false_refusal\|degraded\|benign_degraded\|partial_refusal\|error\)$' "$csvfile" | sort | uniq -c | sort -rn
    
    echo ""
    echo "CORRECTION_APPLIED:"
    # Look for pattern after prompt_idx, before other fields
    # correction_applied is the 4th field - should be Yes or No
    head -n 100 "$csvfile" | grep -o ',Yes,' | wc -l
    echo "Yes: ^^"
    head -n 100 "$csvfile" | grep -o ',No,' | wc -l
    echo "No: ^^"
    
done
