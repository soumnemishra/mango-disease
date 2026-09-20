import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model import MangiferaNet

def main():
    DATA_DIR = "split_data"
    BATCH_SIZE = 32
    checkpoint_path = Path(r"D:\mango-leaf-detection\MODEL\model-weights\checkpoints\best_mango_variety_model_.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found. Train the model first.")
        return

    print("Loading test data...")
    dataloaders, class_names, _ = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    num_classes = len(class_names)

    print("=" * 60)
    print("EVALUATING BEST MODEL ON HELD-OUT TEST SET")
    print("=" * 60)

    # 1. Load the lowest-loss checkpoint
    print("Loading model...")
    model = MangiferaNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating...")
    # 2. Run inference on the hidden test set
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Print Classification Metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # 4. Plot the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Mango Variety Classification - Confusion Matrix (Test Set)")
    plt.xlabel("Predicted Variety")
    plt.ylabel("Ground Truth Variety")
    plt.xticks(rotation=30, ha='right')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()

