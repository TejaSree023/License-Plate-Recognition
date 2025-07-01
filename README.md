# 🚗 License Plate Recognition (LPR) - SoulPage IT Assessment

This project implements an end-to-end License Plate Recognition pipeline using deep learning techniques. The task involves detecting license plates from vehicle images and recognizing the alphanumeric characters they contain.

---

## 📁 Dataset Overview

The provided data consists of three parts:

- **Training Set 1**: 900 vehicle images with bounding box annotations for license plates.
- **Training Set 2**: 900 cropped license plate images with character annotations.
- **Test Set**: 201 vehicle images for final evaluation.

---

## 🧠 Models Used

### 1. 🔍 License Plate Detector
- **Architecture**: CNN using MobileNetV2 backbone.
- **Input**: (224x224x3) image
- **Output**: Normalized bounding box coordinates (ymin, xmin, ymax, xmax)
- **Training File**: `detector_train.py`

### 2. 🔤 License Plate OCR (Character Recognition)
- **Architecture**: CRNN (CNN + BiLSTM + CTC Loss)
- **Input**: (50x200x1) grayscale image
- **Output**: Decoded alphanumeric string
- **Training File**: `ocr_train.py`

---

## 🧪 Inference & Submission

### ✅ Final Inference Script: `test_inference.ipynb`

- **Step 1**: Loads trained detector and OCR models
- **Step 2**: Detects plates in each test image
- **Step 3**: Crops the plate region and runs OCR
- **Step 4**: Creates a CSV with 10-digit indicator columns as required:
