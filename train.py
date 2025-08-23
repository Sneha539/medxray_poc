#!/usr/bin/env python
import argparse, os, json, time, math, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_dataloaders(data_dir, img_size=224, batch_size=16):
    # Expect ImageFolder layout: train/val/test/<class>/image.png
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=os.path.join(data_dir, "val"),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(root=os.path.join(data_dir, "test"),  transform=eval_tfms)

    # Class weights for imbalance
    targets = [y for _, y in train_ds.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    label_map = {i: c for i, c in enumerate(train_ds.classes)}
    return train_loader, val_loader, test_loader, label_map

def build_model(num_classes=2, model_name="densenet121", pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=3)
    return model

def evaluate(model, loader, criterion):
    model.eval()
    all_logits, all_targets = [], []
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            all_logits.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    probs = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(targets, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)

    # --- FIXED AUC ---
    try:
        if probs.shape[1] == 2:  # binary classification
            auc = roc_auc_score(targets, probs[:, 1])
        else:  # multi-class
            auc = roc_auc_score(targets, probs, multi_class="ovr")
    except Exception:
        auc = 0.0   # fallback instead of NaN

    loss = running_loss / len(loader.dataset)
    return {"loss": loss, "acc": acc, "precision": pr, "recall": rc, "f1": f1, "auc": auc}


def bce_loss(num_classes):
    # For multi-class, CrossEntropy is standard; for binary, BCE with logits also works.
    # Here we assume softmax outputs via CrossEntropy because we used num_classes>1.
    return nn.CrossEntropyLoss()

def train(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_loader, val_loader, test_loader, label_map = build_dataloaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size
    )
    with open("models/label_map.json", "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    model = build_model(num_classes=len(label_map), model_name=args.model_name, pretrained=True).to(DEVICE)

    criterion = bce_loss(len(label_map))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_auc = -1.0
    best_acc = 0.0 # in case AUC is not computable
    log_path = "logs/train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss","val_acc","val_precision","val_recall","val_f1","val_auc","lr"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion)

        # log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_metrics['loss']:.4f}", f"{val_metrics['acc']:.4f}",
                             f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}", f"{val_metrics['f1']:.4f}",
                             f"{val_metrics['auc']:.4f}", f"{scheduler.get_last_lr()[0]:.6f}"])

        # save best
    current_auc = val_metrics["auc"]
    current_acc = val_metrics["acc"]

     # save if AUC improved, or if no valid AUC, use accuracy
    if (not math.isnan(current_auc) and current_auc > best_auc) or (math.isnan(current_auc) and current_acc > best_acc):
        best_auc = -1.0 if math.isnan(current_auc) else current_auc
        best_acc = current_acc
        torch.save(model.state_dict(), "models/best_model.pt")

    scheduler.step()

    # final test metrics
    test_metrics = evaluate(model, test_loader, criterion)
    with open("logs/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Test metrics:", test_metrics)
    print("Training complete. Best val AUC:", best_auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder with train/val/test subfolders (ImageFolder layout).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="densenet121")
    args = parser.parse_args()
    train(args)
