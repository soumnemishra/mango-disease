import os
import shutil
from pathlib import Path
from collections import defaultdict

# Point this to your NEW cleaned directory
CLEAN_DIR = Path(r"D:\mango-leaf-detection\cleaned-data-mango-leaf")
QUARANTINE_DIR = Path(r"D:\mango-leaf-detection\quarantine-dir")

file_map = defaultdict(list)
condition_quarantine_count = 0

# Scan the cleaned directory
for file_path in CLEAN_DIR.rglob("*.jpg"):
    file_map[file_path.name].append(file_path)

for filename, paths in file_map.items():
    # If the exact same filename exists in multiple paths inside the clean directory
    if len(paths) > 1:
        for p in paths:
            # Move them to quarantine and remove them from the clean dataset
            safe_name = f"CONDITION_CONFLICT_{p.parent.parent.name}_{p.parent.name}_{filename}"
            shutil.move(str(p), str(QUARANTINE_DIR / safe_name))
            condition_quarantine_count += 1

print(f"Quarantined {condition_quarantine_count} cross-condition duplicates.")