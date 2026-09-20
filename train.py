import os
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import get_dataloaders
from src.model import MangiferaNet

def main():
    EPOCHS = 100
    WEIGHT_DECAY = 1e-2
    BATCH_SIZE = 32
    DATA_DIR = "split_data"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    dataloaders, class_names, image_datasets = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # 2. Compute Class Weights for Imbalance
    print("Computing class weights...")
    train_targets = np.array(image_datasets['train'].targets)
    unique_classes = np.unique(train_targets)
    raw_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_targets)
    clipped_weights = np.clip(raw_weights, a_min=0.5, a_max=1.75)
    class_weights_tensor = torch.tensor(clipped_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 3. Initialize Model
    model = MangiferaNet(num_classes=num_classes).to(device)

    # 4. Differential Optimizer Setup
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  
        {'params': head_params, 'lr': 1e-4}       
    ], weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # 5. Checkpointing logic
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path = checkpoint_dir / "last_checkpoint.pth"
    best_model_path = checkpoint_dir / "best_model.pth"

    start_epoch = 0
    best_val_loss = float('inf')

    if resume_checkpoint_path.exists():
        print(f"Resuming from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

    print(f"Starting training from epoch {start_epoch + 1}")
    start_time = time.time()

    # 6. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc * 100:.2f}%")

        scheduler.step(epoch_val_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
        }, resume_checkpoint_path)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved improved checkpoint to {best_model_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Lowest Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

