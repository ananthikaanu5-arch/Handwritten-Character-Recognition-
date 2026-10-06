"""Draw a character with the mouse and get a live prediction (Tkinter).

    python -m src.app --model models/mnist_cnn.pt
"""
import argparse
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .predict import load_model, predict

SIZE = 280


class App:
    def __init__(self, root, model, classes):
        self.model, self.classes = model, classes
        root.title("Handwritten Character Recognition")
        self.canvas = tk.Canvas(root, width=SIZE, height=SIZE, bg="black", cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.run())
        self.label = tk.Label(root, text="Draw a character", font=("Helvetica", 16))
        self.label.pack()
        tk.Button(root, text="Clear", command=self.clear).pack(pady=8)
        self.img = Image.new("L", (SIZE, SIZE), 0)
        self.d = ImageDraw.Draw(self.img)

    def draw(self, e):
        r = 10
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="white", outline="white")
        self.d.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)

    def run(self):
        arr = np.array(self.img)
        if arr.max() == 0:
            return
        top = predict(self.model, self.classes, arr)
        self.label.config(text="   ".join(f"{c} ({p:.0%})" for c, p in top))

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (SIZE, SIZE), 0)
        self.d = ImageDraw.Draw(self.img)
        self.label.config(text="Draw a character")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    model, classes = load_model(args.model)
    root = tk.Tk()
    App(root, model, classes)
    root.mainloop()


if __name__ == "__main__":
    main()
