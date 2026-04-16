import csv
from pathlib import Path

files = [
    ('assistant_cap_jailbreak', 'Final Results/assistant_cap_jailbreak (2).csv'),
    ('assistant_cap_benign', 'Final Results/assistant_cap_benign (2).csv'),
    ('cross_cap_jailbreak', 'Final Results/cross_cap_jailbreak (3).csv'),
    ('cross_cap_benign', 'Final Results/cross_cap_benign (3).csv'),
]

for name, filepath in files:
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total = len(rows)
    corrections = {'Yes': 0, 'No': 0}
    layers_with_correction = 0
    
    for row in rows:
        correction = row.get('correction_applied', '').strip()
        if correction == 'Yes':
            corrections['Yes'] += 1
            if row.get('layers', '').strip():
                layers_with_correction += 1
        elif correction == 'No':
            corrections['No'] += 1
    
    print(f'\n{name.upper()}:')
    print(f'  Total rows: {total}')
    print(f'  Correction_applied = Yes: {corrections["Yes"]}')
    print(f'  Correction_applied = No: {corrections["No"]}')
    print(f'  With layer data: {layers_with_correction}')
