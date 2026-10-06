"""Train the CNN on MNIST or EMNIST.

Examples:
    python -m src.train --dataset mnist --epochs 10
    python -m src.train --dataset emnist --split balanced --epochs 15
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from .data import get_loaders
from .model import CNN


def evaluate(model, loader, device):
    model.eval()
    correct, total, preds, labels = 0, 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
            preds.append(p.cpu()); labels.append(y.cpu())
    return correct / total, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "emnist"])
    ap.add_argument("--split", default="balanced", help="EMNIST split")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, classes = get_loaders(args.dataset, args.split, args.batch_size)
    model = CNN(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.out, exist_ok=True)
    tag = args.dataset if args.dataset == "mnist" else f"emnist_{args.split}"
    best, history = 0.0, {"loss": [], "acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        sched.step()
        acc, preds, labels = evaluate(model, test_loader, device)
        avg = running / len(train_loader.dataset)
        history["loss"].append(avg); history["acc"].append(acc)
        print(f"epoch {epoch:02d}  loss {avg:.4f}  test acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "dataset": args.dataset,
                        "split": args.split}, os.path.join(args.out, f"{tag}_cnn.pt"))

    # reports
    with open(os.path.join(args.out, f"{tag}_history.json"), "w") as f:
        json.dump(history, f)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["loss"]); ax[0].set_title("Train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(history["acc"]); ax[1].set_title("Test accuracy"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"{tag}_curves.png"), dpi=120)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.log1p(cm), cmap="Blues")
    ax.set_title("Confusion matrix (log scale)")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, f"{tag}_confusion.png"), dpi=120)
    print(f"best test accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
