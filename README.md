# Transfer Learning: Feature Extraction vs Fine-Tuning

## Overview

This project compares two **transfer learning approaches** for image classification using **MobileNetV2 pretrained on ImageNet** and a **five-class flower dataset**:

- **Feature Extraction** — the pretrained base model remains frozen while only a new classification head is trained.
- **Fine-Tuning** — the model is first trained with the base frozen, then the top pretrained layers are unfrozen and trained with a lower learning rate.

The comparison evaluates model performance, training behaviour, per-class F1-scores, and the trade-off between accuracy and computational cost.

---

## Dataset

The project uses a flower image classification dataset with five classes:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

### Image Configuration

- **Image size:** 160 × 160
- **Batch size:** 32
- **Random seed:** 42

Data augmentation was applied during training.

---

## Base Model

Both approaches use **MobileNetV2** with:

- **ImageNet pretrained weights**
- `include_top=False`
- MobileNetV2-specific preprocessing

The same base architecture and shared setup were used to make the comparison consistent.

---

# Approach 1: Feature Extraction

In feature extraction, the pretrained MobileNetV2 base remained **frozen for the entire run**.

Only the classification head was trained:

```text
Input Image
     ↓
Data Augmentation
     ↓
MobileNetV2 Preprocessing
     ↓
Frozen MobileNetV2 Base
     ↓
GlobalAveragePooling2D
     ↓
Dropout
     ↓
Dense Softmax Output Layer
```

### Training Setup

- **Trainable parameters:** 6,405
- **Learning rate:** 1e-3
- **Maximum epochs:** 30
- **Actual epochs trained:** 13
- **Early stopping:** Patience of 5 on validation loss
- **Best weights restored from:** Epoch 8

---

# Approach 2: Fine-Tuning

Fine-tuning was performed in **two phases**.

## Phase 1: Warm-Up

The MobileNetV2 base model remained frozen while the classification head was trained.

- **Maximum epochs:** 30
- **Actual epochs trained:** 13
- **Learning rate:** 1e-3
- **Best validation accuracy:** 87.17%

The training trajectory was nearly identical to feature extraction, confirming that both notebooks had an equivalent setup before unfreezing the base model.

## Phase 2: Fine-Tuning

After warm-up, the top **54 of 154 base layers** were unfrozen.

- **Fine-tuning epochs:** 9 of 20
- **Learning rate:** 1e-5
- **Trainable parameters:** 1,867,845
- **Best validation accuracy:** 90.31%
- **Best validation loss:** 0.2787
- **Best epoch:** Epoch 17

The lower learning rate was used to adapt pretrained features more gradually during fine-tuning.

---

# Results

## Overall Performance

| Metric | Feature Extraction | Fine-Tuning |
|---|---:|---:|
| **Test Accuracy** | 87.50% | **89.49%** |
| **Test Loss** | 0.3685 | **0.3228** |
| **Epochs Trained** | 13 of 30 | 13 + 9 = 22 total |
| **Best Epoch** | Epoch 8 | Epoch 17 |
| **Final Trainable Parameters** | 6,405 | 1,867,845 |
| **Macro Avg F1** | 0.88 | **0.89** |
| **Weighted Avg F1** | 0.88 | **0.89** |

### Key Result

Fine-tuning improved:

- **Test accuracy by +1.99 percentage points**
- **Test loss by 0.0457**
- **Macro and weighted F1-scores from 0.88 to 0.89**

This improvement required approximately **9 additional epochs** and training roughly **290× more parameters** than feature extraction.

---

# Per-Class F1-Score Comparison

| Class | Feature Extraction | Fine-Tuning | Change |
|---|---:|---:|---:|
| Daisy | 0.90 | 0.91 | +0.01 |
| Dandelion | 0.93 | 0.94 | +0.01 |
| Roses | 0.83 | 0.85 | +0.02 |
| Sunflowers | 0.87 | 0.89 | +0.02 |
| Tulips | 0.86 | 0.86 | 0.00 |

### Observation

Every class either **improved or remained unchanged** after fine-tuning. No class regressed in F1-score.

The largest gains were observed for:

- **Roses:** +0.02
- **Sunflowers:** +0.02

Tulips remained unchanged at **0.86 F1-score**.

---

# Precision and Recall Observations

Fine-tuning changed some class decision boundaries rather than improving every metric uniformly.

### Dandelion

- Recall: **0.97 → 0.92**
- Precision: **0.90 → 0.97**

### Sunflowers

- Recall: **0.88 → 0.97**
- Precision: **0.86 → 0.82**

These shifts suggest that fine-tuning changed how the model separated visually similar classes.

---

# Training Curve Highlights

## Feature Extraction

Training accuracy increased from:

**63.9% → 90.4%**

Validation accuracy peaked at:

**86.65% at Epoch 8**

After this point, validation loss began increasing, showing a contained overfitting pattern that early stopping handled by restoring the best weights.

## Fine-Tuning — Phase 1

The warm-up phase followed a very similar trajectory:

- Train accuracy: **63.8% → 88.9%**
- Best validation accuracy: **87.17%**

## Fine-Tuning — Phase 2

After unfreezing the base layers:

- Train accuracy briefly dropped to **77.3%** at the first unfreezing epoch.
- Validation performance then improved beyond the Phase 1 ceiling.
- Best validation accuracy reached **90.31%**.
- Best validation loss reached **0.2787**.

Early stopping ended the full run after **22 total epochs**.

---

# Methodology Comparison

| Aspect | Feature Extraction | Fine-Tuning |
|---|---|---|
| **Base model state** | Frozen for entire run | Frozen during warm-up, then top 54 layers unfrozen |
| **What trains** | Classification head only | Head → head + selected base layers |
| **Trainable parameters** | 6,405 | 1,867,845 |
| **Learning rate** | 1e-3 throughout | 1e-3 → 1e-5 |
| **Training phases** | Single phase | Warm-up + fine-tuning |
| **Early stopping** | Patience 5 on validation loss | Applied independently in each phase |
| **Relative compute cost** | Lower | Higher |
| **Potential failure mode** | Underfitting | Catastrophic forgetting if misconfigured |

---

# Key Takeaway

**Fine-tuning achieved better overall performance than feature extraction in this experiment**, improving test accuracy from **87.50% to 89.49%** while reducing test loss.

However, the improvement came with a higher computational cost:

- **22 total training epochs vs 13**
- **1,867,845 trainable parameters vs 6,405**
- Partial backpropagation through **54 additional MobileNetV2 layers**

This demonstrates the central trade-off in transfer learning: **feature extraction is computationally efficient, while fine-tuning can provide better task-specific performance by allowing pretrained features to adapt to the new dataset.**

---

# Evaluation

Both approaches were evaluated using:

- Training and validation accuracy
- Training and validation loss
- Test accuracy and test loss
- Classification reports
- Macro and weighted F1-scores
- Per-class F1-scores
- Precision and recall
- Confusion matrices
- Sample predictions

---

# Repository Structure

```text
Transfer-Learning-Comparison/
│
├── feature_extraction.ipynb
├── fine_tuning.ipynb
│
├── Transfer_Learning_Comparison_Report.pdf
│
├── README.md
└── requirements.txt
```

---

# Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Learning Outcomes

This project demonstrates practical understanding of:

- Transfer Learning
- Feature Extraction
- Fine-Tuning
- Pretrained Convolutional Neural Networks
- MobileNetV2
- Image Classification
- Data Augmentation
- Early Stopping
- Learning Rate Adjustment
- Classification Metrics
- Precision and Recall Trade-offs
- F1-Score Analysis
- Confusion Matrices
- Training and Validation Curve Analysis

---

## Author

**Ramsha Shahid**  
Data Science Undergraduate | Machine Learning & Deep Learning
