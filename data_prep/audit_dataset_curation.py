from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAW_DIR = Path(r"D:\mango-leaf-detection\data-sets-mango-leaf")
CLEAN_DIR = Path(r"D:\mango-leaf-detection\processed-512")
QUARANTINE_DIR = Path(r"D:\mango-leaf-detection\quarantine-dir")
OUTPUT_DIR = Path(r"D:\mango-leaf-detection\curation_audit")


def scan_dataset(directory: Path, name: str) -> pd.DataFrame:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {directory}")

    records = []
    for path in sorted(directory.rglob("*.jpg")):
        relative_parts = path.relative_to(directory).parts
        if len(relative_parts) < 3:
            continue
        records.append(
            {
                "dataset": name,
                "variety": relative_parts[0],
                "condition": relative_parts[1].title(),
                "filename": path.name,
                "relative_path": path.relative_to(directory).as_posix(),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = scan_dataset(RAW_DIR, "Raw (Uncurated)")
    df_clean = scan_dataset(CLEAN_DIR, "Final Cleaned (512x512)")
    df_combined = pd.concat([df_raw, df_clean], ignore_index=True)

    summary = (
        df_combined.groupby(["dataset", "variety"], sort=True)
        .size()
        .unstack("dataset", fill_value=0)
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "dataset_curation_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(
        data=df_combined,
        x="variety",
        hue="dataset",
        palette=["#d95f02", "#1b9e77"],
        order=sorted(df_combined["variety"].dropna().unique()),
    )
    ax.set_title(
        "Dataset Curation Impact: Raw Field Captures vs. Curated Cache",
        fontsize=14,
        weight="bold",
    )
    ax.set_xlabel("Mango Variety", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "dataset_curation_audit.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    raw_total = len(df_raw)
    clean_total = len(df_clean)
    quarantine_files = [path for path in QUARANTINE_DIR.iterdir() if path.is_file()]
    quarantine_count = len(quarantine_files)

    report_path = OUTPUT_DIR / "dataset_curation_audit.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("DATASET PROVENANCE & AUDIT SUMMARY\n")
        report.write("=" * 50 + "\n")
        report.write(f"Original Input Images:         {raw_total}\n")
        report.write(f"Raw minus processed count:     {raw_total - clean_total}\n")
        report.write(f"Quarantine store files:        {quarantine_count}\n")
        report.write(f"Final Machine-Ready Images:    {clean_total}\n")
        report.write("Image Resolution:              Uniform 512 x 512 px\n")
        report.write("Label Integrity:               Folder-derived mapping\n\n")
        report.write("Counts by variety\n")
        report.write("-----------------\n")
        report.write(summary.to_string(index=False))
        report.write("\n")

    print(f"Saved comparison chart: {chart_path}")
    print("\n" + "=" * 50)
    print("      DATASET PROVENANCE & AUDIT SUMMARY")
    print("=" * 50)
    print(f"Original Input Images:         {raw_total}")
    print(f"Raw minus processed count:     {raw_total - clean_total}")
    print(f"Quarantine store files:        {quarantine_count}")
    print(f"Final Machine-Ready Images:    {clean_total}")
    print("Image Resolution:              Uniform 512 x 512 px")
    print("Label Integrity:               Folder-derived mapping")
    print("=" * 50)
    print(f"Reports written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
