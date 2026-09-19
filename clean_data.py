import os
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ==========================================
# 1. Define Directories
# ==========================================
# Coding Agent: Update SRC_DIR to the exact path of the raw dataset
SRC_DIR = Path(r"D:\mango-leaf-detection\data-sets-mango-leaf")
OUT_DIR = Path(r"D:\mango-leaf-detection\cleaned-data-mango-leaf")
QUARANTINE_DIR = Path(r"D:\mango-leaf-detection\quarantine-dir")

# Create output directories safely
OUT_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Map Filenames to Detect Duplicates
# ==========================================
file_map = defaultdict(list)
image_extensions = ['*.jpg', '*.jpeg', '*.webp']

# Recursively scan for all images and map them by filename
for ext in image_extensions:
    for file_path in SRC_DIR.rglob(ext):
        # file_path.name gets the exact filename (e.g., '20250404_161813.jpg')
        file_map[file_path.name].append(file_path)

print(f"Scanning complete. Found {len(file_map)} unique filenames.")

# ==========================================
# 3. Process, Standardize, and Copy Data
# ==========================================
quarantine_count = 0
processed_count = 0

for filename, paths in file_map.items():
    # The dataset structure is Variety/Condition/filename
    # We extract the variety (parent.parent.name) to check for cross-class duplicates
    varieties = set(p.parent.parent.name for p in paths)
    
    # --- ISOLATE CROSS-CLASS DUPLICATES ---
    if len(varieties) > 1:
        for p in paths:
            # Prepend variety and condition to the filename so they don't overwrite each other in quarantine
            safe_name = f"{p.parent.parent.name}_{p.parent.name}_{filename}"
            shutil.copy2(p, QUARANTINE_DIR / safe_name)
            quarantine_count += 1
        continue # Skip copying these to the clean dataset
        
    # --- STANDARDIZE DIRECTORIES AND FORMATS ---
    for p in paths:
        variety = p.parent.parent.name
        raw_condition = p.parent.name
        
        # Standardize folder naming dynamically
        if "before" in raw_condition.lower():
            condition = "Before Ripening"
        elif "after" in raw_condition.lower():
            condition = "After Ripening"
        else:
            condition = raw_condition # Fallback for unexpected folders
            
        dest_folder = OUT_DIR / variety / condition
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Handle Format Uniformity
        dest_ext = p.suffix.lower()
        if dest_ext in ['.webp', '.jpeg']:
            dest_name = p.stem + ".jpg"
            dest_path = dest_folder / dest_name
            
            # Convert to standard RGB JPEG
            with Image.open(p) as img:
                img.convert('RGB').save(dest_path, 'JPEG')
        else:
            # Direct copy for standard .jpg files
            dest_path = dest_folder / p.name
            shutil.copy2(p, dest_path)
            
        processed_count += 1

print(f"Data triage complete.")
print(f"Successfully copied/converted: {processed_count} images to {OUT_DIR}")
print(f"Quarantined: {quarantine_count} conflicting duplicate images to {QUARANTINE_DIR}")

# ==========================================
# 4. Synchronize Annotations 
# ==========================================
# Because .webp and .jpeg files were renamed to .jpg, the annotations file must be updated.
ann_source = SRC_DIR / "mango-leaf-detections"
ann_dest = OUT_DIR / "mango-leaf-detections"

if ann_source.exists():
    with open(ann_source, 'r', encoding='utf-8') as f:
        ann_data = f.read()
    
    # Replace old extensions globally so bounding boxes match the newly converted images
    ann_data = ann_data.replace('.webp', '.jpg').replace('.jpeg', '.jpg')
    
    with open(ann_dest, 'w', encoding='utf-8') as f:
        f.write(ann_data)
    print(f"Annotation file 'mango-leaf-detections' successfully updated and copied.")
else:
    print(f"Warning: 'mango-leaf-detections' not found at {ann_source}.")