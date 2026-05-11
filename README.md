# Learning–Estimation–Decision LED
*A Self-Supervised Learning–Estimation–Decision Framework for Robust Cell Tracking*

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)

---

## 📌 Introduction

a novel **Self-Supervised Learning–Estimation–Decision (LED)** framework for robust cell tracking in time-lapse microscopy sequences.  

The framework integrates:

- **Learning**: self-supervised representation learning to represent cell movement and division pattens  
- **Estimation**: posterior linking probability based on Bayesian theorem
- **Decision**: global optimization to resolve cell associations, divisions, and disappearances  

These designs enable stable tracking for unseen data under vary imaging conditions, dense cell populations, and vary cell types.

![example_lineage_tracks](example_lineage_tracks.gif)

---

## 🧰 Dependencies

### Conda environment (recommended)

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Pip packages

```bash
pip install tensorboard
pip install tqdm
pip install hydra-core==1.3.2
pip install omegaconf
pip install tifffile
pip install scipy
pip install pandas
pip install scikit-image
pip install opencv-python
pip install ortools
pip install imagecodecs
pip install ete3
pip install scikit-learn
```

### Optional (GUI visualization)

```bash
pip install PyQt5
```

---

## 📁 Data Preparation

Place your data in the following structure:

```text
data/
├── img/        # Time-lapse cell images (*.tif)
│   ├── frame_0001.tif
│   ├── frame_0002.tif
│   └── ...
├── mask/       # Cell segmentation masks (same resolution as images, *.tif)
│   ├── mask_0001.tif
│   ├── mask_0002.tif
│   └── ...
```

✅ Each image must have a corresponding segmentation mask (with sorted name).

---

## ⚙️ Key Parameters

Important tracking-related parameters are defined in the configuration file (`config/config.yaml`).

### Core parameters

| Parameter | Description                                         |
|----------|-----------------------------------------------------|
| `max_movemment` | Maximum allowed **pixel** distance for cell linking |

### Optional parameters

| Parameter | Description |
|----------|-------------|
| `division_detect` | whether to pre-detect division for subsequent self-supervised training |
| `run_num` | number of parallel processes |
| `division` |whether to use division constraints in LP|
| `jitter_thr` | threshold for jitter correction |
| `post_pro` | post process including pruning and merging tracks|
|...|...|

---

## ▶️ Usage Example

### Run cell tracking

```
from train import train_model as tm
from predictor import predictor as pr
from cell_tracking import tracker as ct

tm()

pr()

ct()
```


---

## 📊 Output

After execution, results will be saved as:

```text
results/
├── track.csv                           # Cell trajectory matrix: frame × cell
├── CTC format result                   # Cell Tracking Challenge format 
└── visualization of lieange tree       # Track overlay images
```

---

