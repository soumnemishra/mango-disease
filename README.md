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

Completed training and evaluation:

- Stratified train/validation/test split
- Hybrid Mangifera-Net model implementation
- Model training with class-weighted loss and differential learning rates
- Held-out test-set evaluation
- Classification report and confusion-matrix generation

Remaining work:

- Hyperparameter experiments and ablation studies
- External validation on unseen field data
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

## Model Training and Evaluation

The training pipeline is implemented in `train.py` and uses the locally generated `split_data/` directory. The split is stratified by mango variety with a 70% training, 15% validation, and 15% test allocation using `random_state=42`.

The implemented `Mangifera-Net` architecture combines a MobileNetV2 backbone with custom convolutional and attention-based components. Training includes:

- Class-weighted cross-entropy loss for class imbalance
- AdamW optimization
- Differential learning rates for backbone and custom parameters
- ReduceLROnPlateau scheduling
- Mixed precision when CUDA is available
- Checkpoint saving and resume support

The evaluation pipeline is implemented in `evaluate.py`. The current recorded held-out test result is **90.85% accuracy** on **820 test images**.

Recorded metrics:

| Variety | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Amrapalli | 0.8851 | 0.8415 | 0.8627 | 183 |
| Arka Neelachal Kesari | 0.9605 | 0.9444 | 0.9524 | 180 |
| Banganpalli | 0.9318 | 0.9213 | 0.9266 | 89 |
| Dashehari | 0.8056 | 0.8788 | 0.8406 | 99 |
| Suvarnarekha | 0.9487 | 0.9673 | 0.9579 | 153 |
| Totapuri | 0.8889 | 0.8966 | 0.8927 | 116 |

Overall accuracy: **0.9085**
Macro-average F1: **0.9055**
Weighted-average F1: **0.9087**

## Repository Contents

### Analysis scripts

- `analyze_dataset.py` performs a read-only scan of the original dataset. It reports folder counts, file extensions, image dimensions, unreadable images, naming inconsistencies, and possible duplicate-name groups.
- `analyze_cleaned_dataset.py` analyzes the cleaned dataset. It validates images, records dimensions and file sizes, computes SHA-256 hashes, detects exact duplicate-content groups, and creates plots.
- `audit_dataset_curation.py` compares the original `.jpg` inventory with the final `processed-512` cache and creates a provenance summary and comparison chart.

### Curation and processing scripts

- `clean_data.py` contains the earlier cleaning workflow used for dataset preparation.
- `cleaned_up_.py` performs hash-based duplicate/conflict handling and generates the 512 x 512 processed cache.
- `final_check.py` is an additional duplicate-name checking script. It can move files into quarantine and should be reviewed before execution.
- `data_prep/split_dataset.py` creates the stratified train/validation/test split used by the training pipeline.

### Reports and visualizations

- `analysis_reports/` contains the original inventory report and image inventory CSV.
- `cleaned_analysis_reports/` contains cleaned-dataset reports, CSV metadata, and resolution/file-size plots.
- `curation_audit/` contains the raw-versus-processed comparison chart, summary table, and audit text report.
- `classification_report.txt` contains the held-out test metrics.
- `confusion_matrix.png` contains the held-out test confusion matrix.
- `project_report.tex` contains the formal LaTeX project report.

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

### 4. Create the local data split

```powershell
.\.venv\Scripts\python.exe .\data_prep\split_dataset.py
```

This creates `split_data/` locally. The split contains image files and is intentionally excluded from GitHub.

### 5. Train the model

```powershell
.\.venv\Scripts\python.exe .\train.py
```

Training checkpoints are written locally and are intentionally excluded from GitHub.

### 6. Evaluate the trained model

```powershell
.\.venv\Scripts\python.exe .\evaluate.py
```

The evaluation script writes `classification_report.txt` and `confusion_matrix.png` in the project root.

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
3. Check for near-duplicate or visually identical images across splits.
4. Run ablation studies against a simpler CNN baseline.
5. Evaluate the trained model on an independent external dataset.
6. Export a deployment-ready model and document inference requirements.
7. Record each experiment's random seed, configuration, and results.

## Reproducibility Notes

- Analysis scripts do not modify the dataset.
- Paths in the scripts currently point to the local Windows project directory.
- The reports are snapshots of the dataset state at the time they were generated.
- Re-running the curation process may change counts if the input folders have changed.
- Dataset files are intentionally excluded from GitHub for size, privacy, and reproducibility-control reasons.
- Model checkpoint files are intentionally excluded from GitHub because they are large binary artifacts; the architecture and training code remain version controlled.

## Version Control

The code and reports are maintained in:

https://github.com/soumnemishra/mango-disease

The repository's `.gitignore` excludes:

- `data-sets-mango-leaf/`
- `cleaned-data-mango-leaf/`
- `processed-512/`
- `quarantine-dir/`
- `split_data/`
- `MODEL/model-weights/`
- `.venv/`
- Python cache files
