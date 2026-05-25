# IT9205 — Automated Vehicle Detection and Classification in Traffic Scenes

**Student:** Hatem Isa | **ID:** 202508993
**Programme:** MSc in Artificial Intelligence, Bahrain Polytechnic
**Course:** IT9205 Deep Learning for Computer Vision
**Instructor:** Mr. Ghassan AlShajjar
**Submission Date:** 30 May 2026

---
## Demo Video

**COCO Vehicle Labs.mov:** https://youtu.be/WMPKN1A92TI

Screen recording walkthrough of all four notebooks (Part A, B, C, F) 
with audio narration explaining design decisions, model architecture, 
training results, and evaluation at every stage.

## Annotated Output Video

**traffic_annotated.mp4:** https://youtu.be/FxEahIU3VBM

YOLOv8s applied frame-by-frame to a 30-second traffic video. Colour-coded bounding boxes per class with live vehicle count overlay. 27 FPS, mean 9.3 vehicles per frame.

---

## Project Overview

A complete computer vision pipeline for vehicle classification and detection in real-world traffic scenes, built on the Vehicles-COCO dataset.

**Task 1 — Image Classification:** ResNet50 transfer learning with two-phase training to classify vehicle crops into 4 categories: car, bus, truck, motorcycle. CLAHE contrast enhancement applied as the key preprocessing step.

**Task 2 — Object Detection:** YOLOv8s fine-tuned on Vehicles-COCO to detect and localise all vehicles simultaneously in traffic scene images.

**Video Inference:** Both models applied to a 30-second traffic video (750 frames) producing annotated output footage with per-class colour coding and live vehicle count overlay.

**Extra Work (Part F):**
- F1: Cross-task hierarchical pipeline — YOLOv8s detects bounding boxes, ResNet50 independently classifies each crop. 85.8% agreement rate across 106 detections.
- F2: ONNX export and edge deployment analysis — benchmarked PyTorch CPU vs ONNX CPU, produced accuracy-speed tradeoff chart for roadside camera deployment.

---

## Key Results

| Metric | Value |
|---|---|
| ResNet50 accuracy (with CLAHE) | 75.1% |
| ResNet50 accuracy (without CLAHE) | 73.1% |
| CLAHE ablation gain | +2.0% |
| Weighted F1 score | 0.70 |
| Truck recall (worst class) | 14% |
| YOLOv8s mAP@50 | 0.378 |
| YOLOv8s mAP@50-95 | 0.206 |
| Mean IoU | 0.650 |
| Video inference FPS | 27 |
| Avg vehicles per frame | 9.3 |
| Cross-task agreement rate | 85.8% |
| ONNX CPU speed | 6.4 FPS |
| PyTorch CPU speed | 4.9 FPS |
| ONNX speedup | 1.3x |

---

## Repository Structure

```
├── 202508993_PartA.ipynb                              # Image preprocessing pipeline (CLAHE, HSV, Canny, augmentation)
├── 202508993_PartA_output.pdf                         # Rendered output with visualisations
├── 202508993_PartB.ipynb                              # ResNet50 classifier + YOLOv8s detector + video inference
├── 202508993_PartB_output.pdf
├── 202508993_PartC.ipynb                              # Evaluation, ablation, error analysis, SOTA comparison
├── 202508993_PartC_final.pdf
├── 202508993_PartF.ipynb                              # Cross-task pipeline + ONNX edge deployment
├── 202508993_PartF_final.pdf
├── compressed_traffic_annotated.mp4                   # Compressed annotated output video
├── IT9205_Topic_Proposal_Hatem_Isa_202508993_Approved.pdf
├── DATASET.txt                                        # Dataset download links and Colab setup instructions
├── references.zip                                     # Papers 1-7 (Chang, He, Ioffe, Padilla, Pan, Redmon, Reis)
└── references2.zip                                    # Papers 8-13 (Rishika, Sandler, Shorten, Stainton, Supriya, Yaseen)
```

---

## Dataset

**Vehicles-COCO v1** (Roboflow, 2023) — 18,998 annotated real-world traffic scene images across 4 vehicle classes in YOLO TXT format.

| Property | Value |
|---|---|
| Total Images | 18,998 |
| Classes | Car, Bus, Truck, Motorcycle |
| Split | ~80% train / ~20% validation |
| Format | YOLO TXT |
| Scenes | Urban streets, highways, intersections, parking lots |

**Roboflow:** https://universe.roboflow.com/vehicle-mscoco/vehicles-coco/dataset/1

**Google Drive (pre-downloaded):** https://drive.google.com/drive/folders/1_Hz9Y_uSOp9R5kzDTYFWPer-sU0PpZIw?usp=sharing

**Project resources (trained models, ONNX export, full-quality annotated video):** https://drive.google.com/drive/folders/11hea6LhgtfPQFVFdkjJ1-dK-UKfLNcoY?usp=sharing

---

## Setup

All notebooks run on Google Colab. Mount your Drive and update `DATASET_PATH` at the top of each notebook.

```python
DATASET_PATH = pathlib.Path('/content/drive/MyDrive/.../Vehicles-coco.v1-new-vehicle-dataset.yolov8')
```

Run notebooks in order: **A → B → C → F**

---

## Architecture Summary

**Classifier (Task 1)**
- Backbone: ResNet50 (ImageNet pretrained, frozen in Phase 1)
- Head: GlobalAveragePooling2D → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(4, softmax)
- Phase 1: lr=1e-3, frozen backbone, up to 20 epochs with EarlyStopping
- Phase 2: lr=1e-5, last 30 layers unfrozen, 5 epochs fine-tuning

**Detector (Task 2)**
- Model: YOLOv8s (11.2M params)
- Phase 1: freeze=10, lr=1e-3, 30 epochs
- Phase 2: freeze=0, lr=1e-4, 20 epochs
- Input size: 416×416

---

## Tech Stack

- Python 3.12, TensorFlow 2.20, Ultralytics YOLOv8 8.4.51
- OpenCV, scikit-learn, ONNX Runtime
- Google Colab (CPU — no GPU available during training)
