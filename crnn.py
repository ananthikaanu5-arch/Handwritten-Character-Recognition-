"""Extension: word / sentence recognition with a CRNN (CNN + BiLSTM + CTC).

The CNN extracts features from a text-line image, the BiLSTM models the
character sequence, and CTC loss aligns predictions with the label without
needing per-character segmentation.

Train on any line/word dataset (e.g. IAM) by supplying (image, text) pairs.
A tiny synthetic demo is included:  python -m src.crnn --demo
"""
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0  # CTC blank; characters are 1..len(ALPHABET)
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}


class CRNN(nn.Module):
    def __init__(self, num_classes=len(ALPHABET) + 1, hidden=128):
        super().__init__()
        def block(i, o, pool):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool2d(pool))
            return layers
        self.cnn = nn.Sequential(
            *block(1, 32, (2, 2)),     # H/2,  W/2
            *block(32, 64, (2, 2)),    # H/4,  W/4
            *block(64, 128, (2, 1)),   # H/8,  W/4  (keep width = time axis)
            *block(128, 128, (2, 1)),  # H/16, W/4
        )
        self.rnn = nn.LSTM(128 * 2, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):                      # x: (B,1,32,W)
        f = self.cnn(x)                        # (B,C,H',W')
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out, _ = self.rnn(f)
        return self.fc(out)                    # (B,T,classes)


def ctc_loss(logits, targets, target_lens):
    log_probs = F.log_softmax(logits, 2).permute(1, 0, 2)  # (T,B,C)
    in_lens = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
    return F.ctc_loss(log_probs, targets, in_lens, target_lens, blank=BLANK, zero_infinity=True)


def greedy_decode(logits):
    """Collapse repeats and drop blanks."""
    best = logits.argmax(2).cpu().numpy()
    results = []
    for seq in best:
        out, prev = [], BLANK
        for k in seq:
            if k != prev and k != BLANK:
                out.append(IDX2CHAR[int(k)])
            prev = k
        results.append("".join(out))
    return results


def _demo():
    """Sanity check: forward pass + CTC loss on random data."""
    model = CRNN()
    x = torch.randn(4, 1, 32, 128)
    logits = model(x)
    words = ["hello", "world", "cnn", "ctc"]
    targets = torch.tensor([CHAR2IDX[c] for w in words for c in w])
    lens = torch.tensor([len(w) for w in words])
    print("logits:", tuple(logits.shape), " loss:", float(ctc_loss(logits, targets, lens)))
    print("decoded (untrained):", greedy_decode(logits))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    if ap.parse_args().demo:
        _demo()
