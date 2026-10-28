# Comprehensive Road Scene Understanding for Autonomous Driving

Semantic segmentation and post-hoc anomaly segmentation on road scenes, comparing the
pixel-based **ERFNet** with the mask-based **EoMT** (Encoder-only Mask Transformer).

University course project — Politecnico di Torino, MSc in Data Science and Engineering
(*Artificial Intelligence, Machine Learning and Deep Learning*).
Built on a starting repository provided by the course instructors.

**Team:** Giovanni Orlandi (s355320), Riccardo Polazzi (s361352), Enrico Signorini (s362974), Lisa Vandi (s360247).

Autonomous driving needs perception that works both in familiar scenes and in the presence
of unexpected hazards. Deep segmentation networks are accurate on the classes they were
trained on, but behave poorly on unknown, out-of-distribution (OoD) objects — they tend to
assign them a known label with high confidence. This project studies both sides of that
problem on the same models.

---

## Tasks

**Task 1 — Semantic segmentation.** Evaluate five EoMT configurations on the Cityscapes
validation set and compare them by mIoU. The comparison between the Cityscapes-trained and
the COCO-pretrained checkpoint is non-trivial: there is a **domain gap** (general scenes vs
urban driving) and a **taxonomy gap** (133 COCO categories vs 19 Cityscapes classes).

**Task 2 — Anomaly segmentation.** Keep all models fixed and detect anomalous pixels by
applying post-hoc scoring functions to their outputs, evaluated on five road-anomaly
benchmarks. No anomaly-specific training is used.

---

## Method

### Class-space alignment (COCO → Cityscapes)
Two different alignments are used, because the two tasks consume the model output at
different stages:

- **Task 1 (mIoU):** a lookup table maps each of the 133 COCO categories to its closest
  Cityscapes counterpart; categories with no meaningful equivalent go to a dedicated
  `void` index (19). The table is applied element-wise to the predicted COCO class IDs.
- **Task 2 (anomaly scoring):** scoring operates on dense class-wise scores, so COCO score
  channels belonging to the same Cityscapes class `c` are **summed before** any post-hoc
  function is applied: `f_c^CS(x) = Σ_{k ∈ M(c)} f_k^COCO(x)`. Cityscapes classes with no
  suitable COCO equivalent get zero score.

### Fine-tuning EoMT on Cityscapes (PEFT)
The DINOv2 encoder is always frozen; decoder components are selectively unfrozen. All runs
start from the COCO checkpoint, except the incompatible class head which is randomly
re-initialised for the new class space. Loss: cross-entropy on class logits, binary
cross-entropy and Dice loss on mask logits. Training uses 16-bit AMP; LoRA was not used,
since the setup already involves few trainable parameters. All three configurations run for
15 epochs — all strategies plateau by epoch 5–7, and keeping the budget equal makes the
polynomial LR decay consistent across runs.

| Config | Unfrozen components | Trainable params |
|---|---|---|
| FT-1 | class head only | ~15 K |
| FT-2 | + mask head, upscale modules | ~6.5 M |
| FT-3 | + learnable query embeddings | ~6.6 M |

### Anomaly scoring
ERFNet outputs a per-pixel class score vector directly. EoMT produces `N` masks
`M(x) ∈ [0,1]^{N×H×W}` and per-mask class probabilities `P(x) ∈ [0,1]^{N×K}`; dense semantic
scores are obtained by marginalising over masks:

```
L_k(x)[h,w] = Σ_n  P(x)[n,k] · M(x)[n,h,w]
```

The resulting tensor is then used exactly like ERFNet's per-pixel logits. EoMT inference is
window-based: the image is split into crops, each crop is processed, and the dense scores are
merged back to the original resolution.

| Score | Definition | Applied to |
|---|---|---|
| MSP | `1 − max softmax(z)` | ERFNet, EoMT |
| MaxLogit | `−max z(c)` | ERFNet, EoMT |
| MaxEntropy | `−Σ p(c) log p(c)` | ERFNet, EoMT |
| RbA | `−Σ tanh(z(c))` | EoMT only |
| MSP + temperature | `1 − max softmax(z/T)` | EoMT Cityscapes |

Dense semantic scores are saved after inference and reused across temperature values, so the
forward pass runs only once.

---

## Datasets

**Semantic:** Cityscapes (19 classes; train split for fine-tuning, val split for mIoU).
COCO enters only through the pretrained checkpoint.

**Anomaly:** SMIYC RA-21 and RoadAnomaly (general road anomalies), SMIYC RO-21 (obstacles on
the drivable area), Fishyscapes Lost & Found (real hazards), Fishyscapes Static (Cityscapes
images with synthetic anomalies).

**Metrics:** mIoU for semantic segmentation; AuPRC and FPR95 for anomaly segmentation,
treated as binary pixel-wise classification (anomalous = positive).

---

## Results

### Task 1 — Cityscapes validation

| Model | mIoU (%) |
|---|---|
| EoMT Cityscapes | 81.60 |
| EoMT COCO | 51.79 |
| FT-1 (class head) | 57.60 |
| FT-2 (+ mask head, upscale) | 71.90 |
| FT-3 (+ queries) | 72.30 |

PEFT recovers +20.51 points over the COCO baseline, but FT-3 still sits 9.30 points below the
fully in-domain model: partial adaptation reduces the gap without eliminating it. Most of the
gain comes from unfreezing the mask-generation modules; adapting the queries adds only 0.4.

### Task 2 — Anomaly segmentation

AuPRC (%), higher is better. The full table with FPR95 for every model, method and benchmark
is in the project report.

| Model | Method | SMIYC RA-21 | SMIYC RO-21 | FS L&F | FS Static | Road Anomaly |
|---|---|---|---|---|---|---|
| ERFNet | MSP | 29.10 | 2.71 | 1.75 | 7.47 | 12.42 |
| | MaxLogit | 38.32 | 4.63 | 3.30 | 9.50 | 15.85 |
| | MaxEntropy | 30.97 | 3.04 | 2.58 | 8.88 | 12.67 |
| EoMT Cityscapes | MSP | 68.10 | 94.19 | 16.39 | 58.19 | 72.46 |
| | MaxLogit | 67.49 | 94.23 | 16.38 | 58.37 | 70.77 |
| | MaxEntropy | 68.30 | **94.28** | **19.08** | 56.85 | **74.13** |
| | RbA | 62.70 | 93.52 | 16.51 | 60.56 | 70.14 |
| EoMT COCO | MSP | 70.54 | 90.05 | 11.10 | 85.26 | **77.27** |
| | MaxLogit | 70.96 | 90.68 | 11.45 | 85.60 | 76.86 |
| | MaxEntropy | **71.84** | 90.08 | 11.72 | **86.14** | 76.49 |
| | RbA | 50.17 | 86.87 | 7.29 | 84.76 | 72.99 |
| FT-3 | MSP | 57.45 | 47.38 | 14.71 | 71.42 | 51.31 |
| | MaxLogit | 56.83 | 46.21 | 14.73 | 71.12 | 50.94 |
| | MaxEntropy | 56.86 | 45.21 | 16.89 | 71.61 | 50.99 |
| | RbA | 35.18 | 31.98 | 11.61 | 64.71 | 43.74 |

FPR95 tells a partly different story and matters for a safety-critical setting, since it
measures how many in-distribution pixels are flagged when 95% of the anomalies are caught.
On SMIYC RO-21 the Cityscapes model is both accurate and precise (94.19 AuPRC at 0.37 FPR95),
whereas the COCO model pays for its high AuPRC with a far higher false-positive rate on
RA-21 (50.89) and Road Anomaly (74.35). RbA is the most unstable on this metric: 96.05 FPR95
on RA-21 for the Cityscapes model and 96.08 on FS Static for FT-3.

### Findings
- **Mask-based beats pixel-based.** Every EoMT variant outperforms ERFNet, whose best result
  is 38.32 AuPRC on SMIYC RA-21 and which collapses on road obstacles (4.63 on RO-21). Dense
  per-pixel classification is not sufficient to separate unexpected objects from known
  road-scene classes; predicting regional masks first lets anomalous regions keep mask-level
  structure.
- **In-domain accuracy vs open-world sensitivity.** The COCO-pretrained model is the best on
  diverse or synthetic anomalies (86.14 AuPRC on FS Static with MaxEntropy, vs 56.85 for the
  Cityscapes model), while the Cityscapes model wins on road obstacles (94.28 on RO-21),
  where adaptation to road geometry matters. Fine-tuning on Cityscapes reduces the
  open-world sensitivity inherited from COCO.
- **Fishyscapes Lost & Found is hard for everyone** (best 19.08 AuPRC): small, distant
  hazards often lack clear boundaries and get absorbed into the surrounding road mask.
- **The scoring functions are close.** MSP, MaxLogit and MaxEntropy perform comparably. RbA
  is competitive on the Cityscapes model but degrades on the COCO one, suggesting it is more
  sensitive to the starting point and to domain mismatch.
- **Temperature scaling has virtually no impact.** Over T ∈ [0.5, 5] the MSP-based ranking is
  largely unaffected by rescaling: on SMIYC RA-21 AuPRC moves from 67.94 to 68.55 and FPR95
  stays at 30.4, with comparable flatness on the other benchmarks.

---

## Repository structure

```text
semantic-segmentation-roads/
├── eomt/              EoMT code (model, losses, Lightning modules, datasets, configs) + semantic_eval.py
├── eval/              evalAnomalyEOMT.py, evalAnomalyERFNET.py, temperature.py, ERFNet utilities
├── utils/             model_loading.py, label_mapping.py, data_loading.py
├── trained_models/    ERFNet checkpoints
├── notebooks/
│   ├── step_4/        semantic and panoptic visualisation, COCO vs Cityscapes comparison
│   ├── step_5/        fine-tuning FT-1, FT-2, FT-3 and evaluation
│   ├── step_7/        ERFNet anomaly evaluation
│   └── step_8/        EoMT anomaly evaluation and temperature scaling
├── results/           step_5 (fine-tuning curves), step_8 (anomaly tables, temperature sweep)
└── Anomaly_Segmentation.pdf   project assignment
```

## Setup

```bash
cd eomt
pip install -r requirements.txt
pip install numpy pillow scikit-learn matplotlib pyyaml
```

**Data.** Cityscapes is read directly from `leftImg8bit_trainvaltest.zip` and
`gtFine_trainvaltest.zip`. The anomaly validation sets are expected under one root folder as
`FS_LostFound_full/`, `RoadAnomaly/`, `RoadAnomaly21/`, `RoadObsticle21/`, `fs_static/`, each
with `images/` and `labels_masks/`. EoMT checkpoints are not included in the repository.

## Usage

```bash
cd eval

# EoMT anomaly evaluation (cityscapes | coco | coco_finetuned)
python evalAnomalyEOMT.py --base_dir <Validation_Dataset> --model cityscapes

# ERFNet baseline
python evalAnomalyERFNET.py --base_dir <Validation_Dataset> \
  --loadDir ../trained_models/ --loadWeights erfnet_pretrained.pth --loadModel erfnet.py

# Temperature scaling: save the dense scores once, then sweep T
python evalAnomalyEOMT.py --base_dir <Validation_Dataset> --model cityscapes --save_inf
python temperature.py
```

Semantic evaluation and fine-tuning run through the notebooks in `notebooks/step_4` and
`notebooks/step_5`. See `eval/README.md` for per-script details.

## Reproducibility notes
- Several scripts contain hard-coded Colab/Drive paths (datasets, checkpoints, saved scores,
  outputs) in `eval/evalAnomalyEOMT.py`, `eval/evalAnomalyERFNET.py` and `eval/temperature.py`.
- Seeds are fixed (42) in the evaluation scripts; reported numbers come from single runs.
- ERFNet is evaluated on images resized to 512×1024, EoMT at native resolution with sliding
  windows, so cross-architecture numbers are indicative rather than strictly like-for-like.
- The best fine-tuning checkpoint is selected on the Cityscapes validation set, which is also
  the set used to report mIoU.

## References
Kerssies et al., *Your ViT is Secretly an Image Segmentation Model* (EoMT), CVPR 2025 ·
Cheng et al., *Per-Pixel Classification is Not All You Need* (MaskFormer), NeurIPS 2021 ·
Cheng et al., *Mask2Former*, CVPR 2022 · Oquab et al., *DINOv2*, 2023 ·
Romera et al., *ERFNet*, IEEE T-ITS 2018 · Nayal et al., *RbA*, ICCV 2023 ·
Hendrycks & Gimpel, 2016 · Hendrycks et al., *Scaling OoD Detection*, 2019 ·
Liang et al., *ODIN* (temperature scaling), 2017 · Cordts et al., *Cityscapes*, CVPR 2016 ·
Lin et al., *COCO*, ECCV 2014 · Blum et al., *Fishyscapes*, IJCV 2021 ·
Chan et al., *SegmentMeIfYouCan*, 2021.

## Credits and license
The `eomt/` folder derives from the official EoMT implementation (MIT License, Mobile
Perception Systems Lab, TU/e); the ERFNet evaluation utilities derive from the course template.