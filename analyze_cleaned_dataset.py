from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_CONDITIONS = {"after ripening", "before ripening"}


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(dataset_root: Path) -> tuple[pd.DataFrame, list[str]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    variety_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    for variety_dir in variety_dirs:
        condition_dirs = sorted(path for path in variety_dir.iterdir() if path.is_dir())
        found_conditions = {path.name.casefold() for path in condition_dirs}
        missing_conditions = EXPECTED_CONDITIONS - found_conditions
        if missing_conditions:
            warnings.append(
                f"{variety_dir.name}: missing condition folder(s): "
                + ", ".join(sorted(missing_conditions))
            )

        for condition_dir in condition_dirs:
            image_files = sorted(
                path
                for path in condition_dir.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            )
            if not image_files:
                warnings.append(f"{variety_dir.name}/{condition_dir.name}: no images found")

            for path in image_files:
                status = "ok"
                width = height = None
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                except (OSError, UnidentifiedImageError) as error:
                    status = f"unreadable: {type(error).__name__}"

                rows.append(
                    {
                        "variety": variety_dir.name,
                        "condition": condition_dir.name,
                        "relative_path": path.relative_to(dataset_root).as_posix(),
                        "filename": path.name,
                        "extension": path.suffix.casefold(),
                        "size_bytes": path.stat().st_size,
                        "width": width,
                        "height": height,
                        "aspect_ratio": round(width / height, 4) if width and height else None,
                        "status": status,
                        "sha256": file_hash(path),
                    }
                )

    return pd.DataFrame(rows), warnings


def save_plots(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    counts = (
        data.groupby(["variety", "condition"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    counts.plot(kind="bar", figsize=(13, 7), color=["#2a9d8f", "#e9c46a"])
    plt.title("Images by Mango Variety and Ripening Condition")
    plt.xlabel("Mango variety")
    plt.ylabel("Number of images")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(output_dir / "01_images_by_variety_condition.png", dpi=160)
    plt.close()

    extension_counts = data["extension"].value_counts().sort_values(ascending=False)
    extension_counts.plot(kind="bar", figsize=(8, 5), color="#264653")
    plt.title("Image File Extensions")
    plt.xlabel("Extension")
    plt.ylabel("Number of images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "02_file_extensions.png", dpi=160)
    plt.close()

    valid = data[data["status"] == "ok"].copy()
    valid["megapixels"] = valid["width"] * valid["height"] / 1_000_000
    valid["megapixels"].plot(kind="hist", bins=30, figsize=(9, 5), color="#f4a261", edgecolor="white")
    plt.title("Image Resolution Distribution")
    plt.xlabel("Megapixels")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(output_dir / "03_resolution_distribution.png", dpi=160)
    plt.close()

    valid["size_mb"] = valid["size_bytes"] / (1024 * 1024)
    valid["size_mb"].plot(kind="hist", bins=30, figsize=(9, 5), color="#457b9d", edgecolor="white")
    plt.title("Image File Size Distribution")
    plt.xlabel("File size (MB)")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(output_dir / "04_file_size_distribution.png", dpi=160)
    plt.close()


def save_summary(data: pd.DataFrame, warnings: list[str], output_dir: Path) -> None:
    summary_path = output_dir / "cleaned_dataset_report.txt"
    counts = data.groupby(["variety", "condition"], sort=True).size()
    duplicate_hashes = data[data["status"] == "ok"].groupby("sha256").size()
    duplicate_groups = duplicate_hashes[duplicate_hashes > 1]
    valid = data[data["status"] == "ok"]

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Cleaned mango leaf dataset analysis\n")
        file.write("=" * 37 + "\n\n")
        file.write(f"Total image files: {len(data)}\n")
        file.write(f"Readable image files: {(data['status'] == 'ok').sum()}\n")
        file.write(f"Unreadable image files: {(data['status'] != 'ok').sum()}\n")
        file.write(f"Varieties: {data['variety'].nunique()}\n")
        file.write(f"Condition folders: {data['condition'].nunique()}\n")
        file.write(f"Exact duplicate-content groups: {len(duplicate_groups)}\n\n")

        file.write("Images by variety and condition\n")
        file.write("-------------------------------\n")
        for (variety, condition), count in counts.items():
            file.write(f"{variety} / {condition}: {count}\n")

        file.write("\nImage totals by variety\n")
        file.write("-----------------------\n")
        for variety, count in data["variety"].value_counts().sort_index().items():
            file.write(f"{variety}: {count}\n")

        file.write("\nImage totals by extension\n")
        file.write("-------------------------\n")
        for extension, count in data["extension"].value_counts().sort_index().items():
            file.write(f"{extension}: {count}\n")

        if not valid.empty:
            file.write("\nReadable image dimensions\n")
            file.write("--------------------------\n")
            file.write(f"Smallest width: {int(valid['width'].min())} px\n")
            file.write(f"Largest width: {int(valid['width'].max())} px\n")
            file.write(f"Smallest height: {int(valid['height'].min())} px\n")
            file.write(f"Largest height: {int(valid['height'].max())} px\n")
            file.write(f"Median resolution: {valid['width'].median():.0f} x {valid['height'].median():.0f} px\n")
            file.write(f"Median file size: {valid['size_bytes'].median() / (1024 * 1024):.2f} MB\n")

        file.write("\nWarnings\n--------\n")
        if warnings:
            for warning in warnings:
                file.write(f"- {warning}\n")
        else:
            file.write("None\n")

        if len(duplicate_groups):
            file.write("\nExact duplicate-content groups\n------------------------------\n")
            for digest in duplicate_groups.index:
                paths = data.loc[data["sha256"] == digest, "relative_path"]
                file.write("\n".join(f"- {path}" for path in paths) + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a cleaned mango leaf image dataset and create plots.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path("cleaned-data-mango-leaf"),
        help="Cleaned dataset folder (default: cleaned-data-mango-leaf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cleaned_analysis_reports"),
        help="Output folder (default: cleaned_analysis_reports)",
    )
    args = parser.parse_args()

    data, warnings = scan_dataset(args.dataset_root)
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "cleaned_image_inventory.csv", index=False)
    save_summary(data, warnings, args.output)
    save_plots(data, args.output)

    print(f"Scanned {len(data)} image files.")
    print(f"Readable images: {(data['status'] == 'ok').sum()}")
    print(f"Unreadable images: {(data['status'] != 'ok').sum()}")
    print(f"Reports and plots: {args.output}")


if __name__ == "__main__":
    main()
