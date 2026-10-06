# Handwritten Character Recognition

Identify handwritten digits and characters using image processing and deep learning.

- **Datasets:** MNIST (digits), EMNIST (characters)
- **Model:** Convolutional Neural Network (CNN)
- **Extension:** CRNN (CNN + BiLSTM + CTC) for full word / sentence recognition

## Structure

```
hcr_project/
├── requirements.txt
├── README.md
└── src/
    ├── data.py      # MNIST / EMNIST loaders + augmentation
    ├── model.py     # CNN architecture
    ├── train.py     # training, accuracy curves, confusion matrix
    ├── predict.py   # image preprocessing + inference on your own images
    ├── app.py       # draw-with-mouse GUI (Tkinter)
    └── crnn.py      # sequence model for words/sentences
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
# Digits (about 99%+ test accuracy)
python -m src.train --dataset mnist --epochs 10

# Characters: 47 classes (digits + upper/lowercase merged)
python -m src.train --dataset emnist --split balanced --epochs 15

# Other EMNIST splits: byclass (62), letters (26), digits (10)
```

Outputs go to `models/`: the best checkpoint (`*_cnn.pt`), training curves, and a confusion matrix.

## Predict

```bash
python -m src.predict --model models/mnist_cnn.pt --image my_digit.png
python -m src.app --model models/emnist_balanced_cnn.pt   # draw and predict live
```

`predict.py` handles real-world images: it inverts dark-on-light writing,
thresholds with Otsu, crops to the character, and centers it in a 28x28 canvas
to match MNIST/EMNIST formatting.

## How it works

1. **Image processing:** grayscale, binarize, crop, pad to square, resize, normalize.
2. **CNN:** 4 conv layers (with BatchNorm) in 2 blocks, then a dense classifier with dropout.
3. **Training:** Adam, StepLR schedule, light affine augmentation (rotation, shift, scale).

## Extending to words and sentences

`src/crnn.py` implements a CRNN: the CNN extracts features along the width of a
text-line image, a BiLSTM models the sequence, and CTC loss learns the alignment
without character segmentation. Check it runs with:

```bash
python -m src.crnn --demo
```

To train it for real, feed (line image, transcription) pairs from a dataset such as IAM Handwriting.

## Notes

- EMNIST "balanced" merges confusable upper/lowercase letters (e.g. C/c, O/o), so expect ~88-90% accuracy there versus 99%+ on MNIST.
- Datasets download automatically on first run to `data/`.
