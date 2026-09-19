from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_CONDITIONS = {"after ripening", "before ripening"}
DUPLICATE_SUFFIX_PATTERN = re.compile(r"\s*\(\d+\)(?=\.[^.]+$)")


def inspect_image(path: Path) -> tuple[str, str]:
    """Return image status and dimensions without changing the file."""
    if importlib.util.find_spec("PIL") is None:
        return "not_checked_pillow_missing", ""

    try:
        image_module = importlib.import_module("PIL.Image")

        with image_module.open(path) as image:
            image.verify()
        with image_module.open(path) as image:
            return "ok", f"{image.width}x{image.height}"
    except Exception as error:  # Pillow can raise several format-specific errors.
        return f"unreadable: {type(error).__name__}", ""


def scan_dataset(dataset_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    warnings: list[str] = []

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

    variety_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if not variety_dirs:
        warnings.append("No variety folders were found directly under the dataset folder.")

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
            files = sorted(path for path in condition_dir.rglob("*") if path.is_file())
            image_files = [path for path in files if path.suffix.casefold() in IMAGE_EXTENSIONS]
            non_image_files = [path for path in files if path.suffix.casefold() not in IMAGE_EXTENSIONS]

            if not image_files:
                warnings.append(f"{variety_dir.name}/{condition_dir.name}: no image files found")

            for path in image_files:
                status, dimensions = inspect_image(path)
                rows.append(
                    {
                        "variety": variety_dir.name,
                        "condition": condition_dir.name,
                        "relative_path": path.relative_to(dataset_root).as_posix(),
                        "filename": path.name,
                        "extension": path.suffix.casefold(),
                        "size_bytes": str(path.stat().st_size),
                        "dimensions": dimensions,
                        "status": status,
                    }
                )

            if non_image_files:
                warnings.append(
                    f"{variety_dir.name}/{condition_dir.name}: "
                    f"{len(non_image_files)} non-image file(s) found"
                )

            if condition_dir.name not in {"After Ripening", "Before Ripening"}:
                expected_name = "After Ripening" if condition_dir.name.casefold() == "after ripening" else "Before Ripening"
                warnings.append(
                    f"Condition folder capitalization differs from the standard name: "
                    f"{condition_dir.relative_to(dataset_root).as_posix()} "
                    f"(suggested: {expected_name})"
                )

    return rows, warnings


def write_reports(output_dir: Path, rows: list[dict[str, str]], warnings: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "image_inventory.csv"
    report_path = output_dir / "dataset_report.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["relative_path"])
        writer.writeheader()
        writer.writerows(rows)

    by_folder = Counter(f"{row['variety']} / {row['condition']}" for row in rows)
    by_variety = Counter(row["variety"] for row in rows)
    by_extension = Counter(row["extension"] for row in rows)
    by_status = Counter(row["status"] for row in rows)
    duplicate_like = defaultdict(list)
    for row in rows:
        normalized_name = DUPLICATE_SUFFIX_PATTERN.sub("", row["filename"]).casefold()
        duplicate_like[normalized_name].append(row["relative_path"])
    duplicate_groups = [paths for paths in duplicate_like.values() if len(paths) > 1]

    with report_path.open("w", encoding="utf-8") as file:
        file.write("Mango leaf dataset analysis\n")
        file.write("=" * 28 + "\n\n")
        file.write(f"Total image files: {len(rows)}\n")
        file.write(f"Variety folders: {len(by_variety)}\n")
        file.write(f"Condition folders with images: {len(by_folder)}\n")
        file.write(f"Possible duplicate-name groups: {len(duplicate_groups)}\n\n")

        file.write("Images by variety\n")
        file.write("-----------------\n")
        for name, count in sorted(by_variety.items()):
            file.write(f"{name}: {count}\n")

        file.write("\nImages by variety and condition\n")
        file.write("-------------------------------\n")
        for name, count in sorted(by_folder.items()):
            file.write(f"{name}: {count}\n")

        file.write("\nFile extensions\n")
        file.write("---------------\n")
        for extension, count in sorted(by_extension.items()):
            file.write(f"{extension}: {count}\n")

        file.write("\nImage status\n")
        file.write("------------\n")
        for status, count in sorted(by_status.items()):
            file.write(f"{status}: {count}\n")

        file.write("\nWarnings\n")
        file.write("--------\n")
        if warnings:
            for warning in warnings:
                file.write(f"- {warning}\n")
        else:
            file.write("None\n")

        if duplicate_groups:
            file.write("\nPossible duplicate-name groups\n")
            file.write("------------------------------\n")
            for paths in sorted(duplicate_groups):
                file.write("\n".join(f"- {path}" for path in sorted(paths)))
                file.write("\n\n")

    print(f"Scanned {len(rows)} image files.")
    print(f"Report: {report_path}")
    print(f"Inventory: {csv_path}")
    if warnings:
        print(f"Warnings recorded: {len(warnings)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the mango leaf image dataset without modifying it.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path("data-sets-mango-leaf"),
        help="Path to the dataset root (default: data-sets-mango-leaf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_reports"),
        help="Folder for the generated report files (default: analysis_reports)",
    )
    args = parser.parse_args()
    rows, warnings = scan_dataset(args.dataset_root)
    write_reports(args.output, rows, warnings)


if __name__ == "__main__":
    main()
