import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    source_dir = Path("processed-512")
    output_dir = Path("split_data")
    
    if not source_dir.exists():
        print(f"Error: {source_dir} not found.")
        return

    print("Gathering files...")
    all_files = []
    labels = []
    
    # Map every file to its Mango Variety
    for img_path in source_dir.rglob("*.jpg"):
        # variety is the parent of the parent (e.g., Amrapalli)
        variety = img_path.parent.parent.name
        all_files.append(img_path)
        labels.append(variety)

    print(f"Found {len(all_files)} images.")
    if len(all_files) == 0:
        return

    # First split: 70% Train, 30% Temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_files, labels, test_size=0.30, stratify=labels, random_state=42
    )

    # Second split: Divide the 30% Temporary perfectly in half (15% Val, 15% Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Split distribution: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")

    def copy_to_local(file_paths, split_name):
        print(f"Copying {split_name} split...")
        for img_path in tqdm(file_paths, desc=split_name):
            variety = img_path.parent.parent.name
            
            # Create structure: split_data/train/Amrapalli/
            dest_folder = output_dir / split_name / variety
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy the image file
            shutil.copy2(img_path, dest_folder / img_path.name)

    # Execute the transfer
    copy_to_local(X_train, 'train')
    copy_to_local(X_val, 'val')
    copy_to_local(X_test, 'test')

    print("\nSuccess! Data is securely split locally.")

if __name__ == "__main__":
    main()

