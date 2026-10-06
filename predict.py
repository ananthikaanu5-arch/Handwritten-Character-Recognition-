"""Predict a handwritten character from an image file.

    python -m src.predict --model models/mnist_cnn.pt --image my_digit.png
"""
import argparse

import cv2
import numpy as np
import torch

from .data import MEAN, STD
from .model import CNN


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = CNN(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"]


def preprocess(img_gray):
    """Grayscale image (any size) -> normalized 1x1x28x28 tensor.

    Handles dark-on-light paper by inverting, crops to the character,
    pads to a square, and resizes to 20x20 centered in 28x28 (MNIST style).
    """
    img = img_gray.copy()
    if img.mean() > 127:          # white background -> invert
        img = 255 - img
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        raise ValueError("no ink found in image")
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape
    s = max(h, w)
    square = np.zeros((s, s), np.uint8)
    square[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    small = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    canvas[4:24, 4:24] = small
    t = torch.from_numpy(canvas).float().div(255)
    t = (t - MEAN) / STD
    return t.unsqueeze(0).unsqueeze(0)


def predict(model, classes, img_gray, topk=3, device="cpu"):
    x = preprocess(img_gray).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    p, idx = probs.topk(topk)
    return [(classes[i], float(v)) for v, i in zip(p, idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    model, classes = load_model(args.model)
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    for label, p in predict(model, classes, img):
        print(f"{label}: {p:.1%}")


if __name__ == "__main__":
    main()
