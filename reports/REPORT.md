# Blood-cell detection with YOLO26: report

## Dataset (TXL-PBC)

1260 images, 18143 annotated cells, 3 classes. No missing or invalid labels.

| Split | Images | WBC | RBC | Platelets |
|---|---:|---:|---:|---:|
| train | 882 | 908 | 11220 | 382 |
| val | 252 | 257 | 3383 | 112 |
| test | 126 | 133 | 1699 | 49 |

RBCs make up about 90% of the annotations and platelets about 3%, so the classes are heavily imbalanced.
Figures: [reports/dataset/](dataset/).

## Training

- Model: `yolo26s.pt` (COCO-pretrained), fine-tuned on a Kaggle Tesla T4 GPU
- Image size 640, batch 16, up to 100 epochs (early-stopping patience 25), cosine LR
- Ran 100 epochs in 30.1 min, including the final evaluation
- Best validation mAP50-95: 0.870, at epoch 88

Figures: [reports/training/](training/).

## Model performance (`models/best.pt`)

### Test split (held out, never used for training or model selection)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| WBC | 1.000 | 0.999 | 0.995 | 0.891 |
| RBC | 0.978 | 0.934 | 0.991 | 0.907 |
| Platelets | 0.958 | 0.940 | 0.986 | 0.823 |
| **All** | **0.979** | **0.958** | **0.990** | **0.874** |

### Validation split

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| WBC | 0.989 | 0.996 | 0.994 | 0.908 |
| RBC | 0.955 | 0.966 | 0.990 | 0.909 |
| Platelets | 0.914 | 0.938 | 0.972 | 0.794 |
| **All** | **0.953** | **0.967** | **0.985** | **0.870** |

Inference speed on the T4: 12.2 ms per image.
Confusion matrices and PR curves: [reports/model/](model/).
