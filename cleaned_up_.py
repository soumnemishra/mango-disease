import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageFile
from tqdm import tqdm

CLEAN_DIR = Path(r"D:\mango-leaf-detection\cleaned-data-mango-leaf")
QUARANTINE_DIR = Path(r"D:\mango-leaf-detection\quarantine-dir")
RESIZED_DIR = Path(r"D:\mango-leaf-detection\processed-512")

TARGET_SIZE = (512, 512)

QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
RESIZED_DIR.mkdir(parents=True, exist_ok=True)

def compute_hash(filepath: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

# =======================================================
# STEP A: Hash-Based Deduplication
# =======================================================
print("Indexing image hashes across cleaned dataset...")
hash_map = defaultdict(list)
all_images = list(CLEAN_DIR.rglob("*.jpg"))

for img_path in tqdm(all_images, desc="Hashing"):
    img_hash = compute_hash(img_path)
    hash_map[img_hash].append(img_path)

cross_class_quarantined = 0
intra_class_removed = 0

for img_hash, paths in hash_map.items():
    if len(paths) <= 1:
        continue

    # Variety is parent.parent.name, Condition is parent.name
    varieties = set(p.parent.parent.name for p in paths)
    conditions = set(p.parent.name for p in paths)

    # 1. Contradiction across Varieties or Conditions -> Quarantine all copies
    if len(varieties) > 1 or len(conditions) > 1:
        for p in paths:
            safe_name = f"HASH_CONFLICT_{p.parent.parent.name}_{p.parent.name}_{p.name}"
            shutil.move(str(p), str(QUARANTINE_DIR / safe_name))
            cross_class_quarantined += 1
    else:
        # 2. Intra-class duplicates -> Keep the first, delete redundant copies
        for duplicate_path in paths[1:]:
            duplicate_path.unlink()
            intra_class_removed += 1

print(f"Purged {cross_class_quarantined} contradictory cross-class images to quarantine.")
print(f"Removed {intra_class_removed} redundant intra-class duplicate images to prevent data leakage.")

# =======================================================
# STEP B: Generate High-Speed Pre-Resized Dataset
# =======================================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("\nGenerating pre-resized 512x512 training cache...")
surviving_images = list(CLEAN_DIR.rglob("*.jpg"))
print(f"Resuming resize for {len(surviving_images)} candidate images...")
corrupted_count = 0

for img_path in tqdm(surviving_images, desc="Resizing"):
    variety = img_path.parent.parent.name
    condition = img_path.parent.name

    out_folder = RESIZED_DIR / variety / condition
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / img_path.name

    if out_path.exists():
        continue

    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img_resized.save(out_path, format="JPEG", quality=95)
    except Exception as error:
        safe_name = f"CORRUPT_{variety}_{condition}_{img_path.name}"
        shutil.move(str(img_path), str(QUARANTINE_DIR / safe_name))
        corrupted_count += 1
        print(f"\nQuarantined {img_path}: {error}")

print(f"Resize complete! Quarantined {corrupted_count} unreadable images.")
print(f"Pre-resized cache successfully created at: {RESIZED_DIR}")