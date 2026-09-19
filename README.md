# Mango Leaf Detection Dataset Curation

## Project Status

This repository documents the preparation and quality analysis of a mango leaf image dataset before machine-learning model training.

Completed:

- Dataset folder inspection and inventory
- Image format and readability validation
- Per-variety and per-condition image counting
- Detection of inconsistent condition-folder capitalization
- Detection of possible duplicate filenames
- Exact duplicate-content analysis using SHA-256 hashes
- Dataset curation and conflict quarantine
- Resizing of surviving images to a uniform 512 x 512 JPEG format
- Resumable processing for interrupted resize jobs
- Raw-versus-processed dataset audit and visualization
- Reproducible dependency list and GitHub documentation

Not completed yet:

- Train/validation/test split
- Model selection and training
- Hyperparameter tuning
- Evaluation metrics and confusion matrix
- Final model export and deployment

## Dataset Organization

The dataset is organized by mango variety and ripening condition:

```text
<dataset-root>/
    Amrapalli/
        After Ripening/
        Before Ripening/
    Arka Neelachal Kesari/
        After Ripening/
        Before Ripening/
    Banganpalli/
        After Ripening/
        Before Ripening/
    Dashehari/
        After Ripening/
        Before Ripening/
    Suvarnarekha/
        After Ripening/
        Before Ripening/
    Totapuri/
        After Ripening/
        Before Ripening/
```

The original data contains six mango varieties and two ripening-condition labels. Some original folder names used different capitalization, such as `Before ripening` and `before ripening`. The analysis scripts compare condition names case-insensitively and report naming inconsistencies instead of silently hiding them.

## Verified Dataset Results

### Original inventory

The first read-only inventory scanned **6,896 image files** across the original dataset. All scanned images passed Pillow validation.

The original inventory contained:

- 6,747 `.jpg` files
- 145 `.jpeg` files
- 4 `.webp` files
- 6 mango varieties
- 12 condition folders
- 722 possible duplicate-name groups

The raw-versus-processed audit intentionally counts only `.jpg` files because the processed cache is JPEG-only. Therefore, its raw count is **6,747**, not 6,896.

### Cleaned dataset analysis

Before the final curation script was run, the cleaned dataset analysis reported:

- 5,650 images
- 5,650 readable images
- 0 unreadable images
- 101 exact duplicate-content groups
- Image dimensions ranging from 280 x 289 to 4,608 x 4,608 pixels
- Median source resolution of approximately 4,608 x 2,128 pixels
- Median source file size of approximately 2.46 MB

### Final processed cache audit

The final audit reported:

- Raw `.jpg` images: **6,747**
- Final processed 512 x 512 `.jpg` images: **5,461**
- Raw-minus-processed difference: **1,286**
- Files currently stored in the quarantine directory: **1,170**

The raw-minus-processed difference and the quarantine count are reported separately. They are not assumed to be identical because the cleanup process can remove intra-class duplicates, quarantine cross-class conflicts, and create processed outputs from the surviving files.

Final processed counts by variety:

| Variety | Raw JPG | Final 512 x 512 |
|---|---:|---:|
| Amrapalli | 1,255 | 1,218 |
| Arka Neelachal Kesari | 1,213 | 1,198 |
| Banganpalli | 739 | 592 |
| Dashehari | 698 | 664 |
| Suvarnarekha | 1,961 | 1,016 |
| Totapuri | 881 | 773 |
| **Total** | **6,747** | **5,461** |

The class distribution is not perfectly balanced. Suvarnarekha has substantially more raw images than the other varieties, while the final processing stage removes many duplicate or conflicting files from that class. This imbalance should be considered when creating the training split and interpreting evaluation metrics.

## Repository Contents

### Analysis scripts

- `analyze_dataset.py` performs a read-only scan of the original dataset. It reports folder counts, file extensions, image dimensions, unreadable images, naming inconsistencies, and possible duplicate-name groups.
- `analyze_cleaned_dataset.py` analyzes the cleaned dataset. It validates images, records dimensions and file sizes, computes SHA-256 hashes, detects exact duplicate-content groups, and creates plots.
- `audit_dataset_curation.py` compares the original `.jpg` inventory with the final `processed-512` cache and creates a provenance summary and comparison chart.

### Curation and processing scripts

- `clean_data.py` contains the earlier cleaning workflow used for dataset preparation.
- `cleaned_up_.py` performs hash-based duplicate/conflict handling and generates the 512 x 512 processed cache.
- `final_check.py` is an additional duplicate-name checking script. It can move files into quarantine and should be reviewed before execution.

### Reports and visualizations

- `analysis_reports/` contains the original inventory report and image inventory CSV.
- `cleaned_analysis_reports/` contains cleaned-dataset reports, CSV metadata, and resolution/file-size plots.
- `curation_audit/` contains the raw-versus-processed comparison chart, summary table, and audit text report.

The dataset folders, processed image cache, quarantine directory, virtual environment, and Python cache files are excluded through `.gitignore`. The GitHub repository contains code and analysis artifacts, not the image data.

## Environment Setup

The project uses Python 3.11 and a local virtual environment.

From PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, enable scripts for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

To leave the environment:

```powershell
deactivate
```

The VS Code interpreter should be set to:

```text
D:\mango-leaf-detection\.venv\Scripts\python.exe
```

## Reproducing the Analysis

### 1. Analyze the original dataset

```powershell
.\.venv\Scripts\python.exe .\analyze_dataset.py
```

Default input:

```text
data-sets-mango-leaf
```

Default outputs:

```text
analysis_reports/dataset_report.txt
analysis_reports/image_inventory.csv
```

A different input or output location can be supplied:

```powershell
.\.venv\Scripts\python.exe .\analyze_dataset.py `
    D:\path\to\dataset `
    --output D:\path\to\report
```

### 2. Analyze the cleaned dataset and create plots

```powershell
.\.venv\Scripts\python.exe .\analyze_cleaned_dataset.py
```

Default outputs:

```text
cleaned_analysis_reports/cleaned_dataset_report.txt
cleaned_analysis_reports/cleaned_image_inventory.csv
cleaned_analysis_reports/01_images_by_variety_condition.png
cleaned_analysis_reports/02_file_extensions.png
cleaned_analysis_reports/03_resolution_distribution.png
cleaned_analysis_reports/04_file_size_distribution.png
```

### 3. Run the curation audit

```powershell
.\.venv\Scripts\python.exe .\audit_dataset_curation.py
```

Outputs:

```text
curation_audit/dataset_curation_audit.png
curation_audit/dataset_curation_audit.txt
curation_audit/dataset_curation_summary.csv
```

The chart compares raw field-capture counts with the final 512 x 512 processed cache for each variety.

## Curation Method

The curation process uses the following logic:

1. Compute file hashes to identify byte-identical files.
2. Detect files whose identical content appears under different varieties or conditions.
3. Move cross-label conflicts to `quarantine-dir` for auditability.
4. Remove redundant intra-class duplicate copies.
5. Attempt to load and convert surviving images to RGB.
6. Resize images to 512 x 512 pixels using Pillow's LANCZOS resampling.
7. Save processed images as JPEG files with quality 95.
8. Skip outputs that already exist so an interrupted resize can resume.
9. Enable Pillow's truncated-image loading and quarantine files that still fail completely.

The processed images should be treated as a machine-ready cache. The original and cleaned datasets remain local and are not included in this repository.

## Important Execution Warning

`cleaned_up_.py` contains file-moving and file-deletion operations in its duplicate-handling stage. Before running it on a new dataset, verify the input paths and make a backup. The resize stage is resumable, but rerunning the whole script also reruns the curation stage.

For a future safer version, the curation stage should support a dry-run mode and write a manifest before moving or deleting files.

## Planned Next Steps

1. Review the quarantined conflicts and decide whether any can be restored.
2. Normalize condition folder names consistently.
3. Create a stratified train/validation/test split by variety and ripening condition.
4. Check for near-duplicate or visually identical images across splits.
5. Establish a baseline model, such as transfer learning with a pretrained CNN.
6. Track accuracy, precision, recall, F1 score, and per-class confusion matrices.
7. Evaluate class imbalance and consider weighted loss or balanced sampling.
8. Record the random seed, split manifest, model configuration, and experiment results.

## Reproducibility Notes

- Analysis scripts do not modify the dataset.
- Paths in the scripts currently point to the local Windows project directory.
- The reports are snapshots of the dataset state at the time they were generated.
- Re-running the curation process may change counts if the input folders have changed.
- Dataset files are intentionally excluded from GitHub for size, privacy, and reproducibility-control reasons.

## Version Control

The code and reports are maintained in:

https://github.com/soumnemishra/mango-disease

The repository's `.gitignore` excludes:

- `data-sets-mango-leaf/`
- `cleaned-data-mango-leaf/`
- `processed-512/`
- `quarantine-dir/`
- `.venv/`
- Python cache files
