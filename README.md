# FFNet vehicle-counting reproduction

This project keeps the authors' FFNet implementation intact in
`upstream/Fuss-Free-structure-main/` and adds a small vehicle-data adapter in
`vehicle_ffnet/`. The upstream repository is
[erdongsanshi/Fuss-Free-structure](https://github.com/erdongsanshi/Fuss-Free-structure)
for *The Effectiveness of a Simplified Model Structure for Crowd Counting*
(IEEE TIM, 2025). The downloaded `ffnet-main.zip` SHA-256 is
`50D0F714E8A9A56C1265E1337BF3EF1F0C8B97B0A76A5A4150B236DEAAA66DEA`.
This is a vehicle-domain reproduction of the architecture and loss, not a
reproduction of the published crowd-dataset scores.

## Layout

```text
upstream/Fuss-Free-structure-main/    unchanged authors' code
vehicle_ffnet/                         CARPK/PUCPR+ adapter, train, evaluate
configs/                               reproducible starting settings
data/raw/CARPK/                         local official images and annotations
data/manifests/CARPK/                   NTU validation, 827/162/459
data/manifests/CARPK_GF2_val/           GF2 candidate, 834/155/459
data/manifests/PUCPR+/                  official 100/25, grouped 75/25 train/val
tests/                                 split, mass, stitching checks
outputs/                               checkpoints and per-image results
```

The raw images and manifests are now both under this project's `data/`
directory. Manifests store relative image and annotation names, so images
are not duplicated. Run commands from the project root; on Linux the same
relative paths work after copying this `data/` directory.

## What is reproduced

The unchanged `Networks/FFNet.py` uses pretrained ConvNeXt-Tiny feature maps at
strides 8, 16 and 32 (192, 384 and 768 channels). Each map passes through its
original channel/spatial attention and ODConv-based CCSM block. Two transpose
convolutions align the coarser maps to stride 8; concatenation and a 1x1 ReLU
head produce a nonnegative density map. The map is normalized internally for
the official OT loss. Prediction is the sum of density mass, with official
overlapping-patch stitching for full-size images.

The instantiated network has 34,579,012 parameters. It is a teacher
baseline, not yet a lightweight student model.

Training retains the authors' losses and coefficients:

`L = 0.1 * L_OT + L1(sum(D), number_of_points) + 0.01 * L_TV`.

The adapter reads CARPK/PUCPR+ bounding boxes, uses their **in-image box
centers** as point targets, and bins each accepted crop point into a stride-8
discrete map with unit mass. It does not create a Gaussian GT, relocate
truncated vehicles to hypothetical full-car centers, add a density MSE, or
insert a new module. Random crops and horizontal flips follow the upstream
training pattern. The adapter uses exclusive crop bounds to avoid assigning a
border point twice.

## Data protocol

| Dataset | Train | Validation | Official test | Train plus validation |
| --- | ---: | ---: | ---: | ---: |
| CARPK images | 827 | 162 | 459 | 989 |
| CARPK vehicles | 35,502 | 6,772 | 47,500 | 42,274 |
| CARPK GF2 candidate images | 834 | 155 | 459 | 989 |
| CARPK GF2 candidate vehicles | 35,946 | 6,328 | 47,500 | 42,274 |
| PUCPR+ images | 75 | 25 | 25 | 100 |
| PUCPR+ vehicles | 10,442 | 2,554 | 3,920 | 12,996 |

CARPK reuses the existing fixed split: validation is the held-out
`20161029_NTU` sequence; the 459 test images remain separate. PUCPR+ has an
official 100-image train and 25-image test list but no local validation list.
For model selection we reserve the official-train scene indices 25–33 as
validation; indices 0–24 form training. All weather versions of one numbered
scene remain together, preventing near-duplicate train/validation leakage.
The alternative GF2 candidate puts the complete `20161030_GF2` sequence in
validation and returns the old NTU validation sequence to training. The
official CARPK test list stays exactly the same. CeDiRNet reports a 155-image
same-sequence CARPK validation set; the paper does not name its exact video
sequence. The 155-image GF2 candidate matches that size, but is our explicit
local protocol. Keep the NTU and GF2 results separate: their validation MAEs
are measured on different images. The test split is used only after settings
and checkpoint selection.

Regenerate and audit the manifests if the source data changes:

```powershell
python -m vehicle_ffnet.prepare `
  --carpk-root data/raw/CARPK/CARPK_devkit/data `
  --carpk-legacy-manifests data/manifests/CARPK `
  --pucpr-root data/raw/CARPK/PUCPR+_devkit/data `
  --output-dir data/manifests
```

The GF2 candidate is generated independently from the official CARPK train
and test lists, without changing the NTU manifests:

```powershell
python -m vehicle_ffnet.scene_split
```

Its output is `data/manifests/CARPK_GF2_val/`. The command refuses to
overwrite an existing candidate directory; remove it deliberately before
regenerating a new version.

## Preprocessing compared with the two source projects

Our FFNet vehicle adapter preserves the authors' essential **custom-dataset**
pipeline: image RGB conversion, 256px random training crops, 0.5 horizontal
flip, ImageNet mean/std normalization, one unit-mass point per accepted box
center on the stride-8 grid, and overlapping-patch full-image evaluation.
The discrete target is constructed before horizontal flipping, then flipped
with the image, as in their `CustomDataset`. The custom adapter reads official
vehicle boxes instead of preconverted point files. Crop membership uses
half-open pixel bounds (`0 <= x,y < 256`) rather than the upstream inclusive
upper bound, which could include a point just outside a crop. All local
images are larger than the crop, so upstream's small-image resizing path is
not used. Training schedules and batch size are vehicle/GPU adaptations.

[CeDiRNet](https://github.com/vicoslab/CeDiRNet) instead trains a
**center-direction** field and locates its peaks; it does not train FFNet's
density integral. Its CARPK/PUCPR+ config keeps full images at scale 1.0,
derives surrogate instance regions from each center using a fixed 30px box
during training, and uses horizontal and vertical flips plus color jitter,
each with probability 0.5. It uses `ToTensor` without ImageNet normalization.
For CARPK testing, its config pads symmetrically by 64px and to a multiple
of 32, marks padded regions for special handling, and uses a 20px surrogate
box for evaluation. Those box sizes serve CeDiRNet's direction/instance
targets, not a Gaussian density kernel. Its reported CARPK validation fold
holds out one entire video sequence (155 images). We borrow that **split
principle** for the GF2 candidate, not its augmentation, target, or network.

## Environment and settings

Use a matched PyTorch/torchvision pair for the installed CUDA driver. The
upstream `requirements.txt` combines incompatible torch/torchvision pins; do
not install that file as-is. The adapter was checked with torch 2.7.1+cu126
and torchvision 0.22.1+cu126. Install the remaining small dependencies from
`requirements-adapter.txt`. Real training needs the upstream ConvNeXt-Tiny
ImageNet V1 weights; torchvision will download them on first use unless
already cached. `--no-pretrained` is only for offline smoke tests.

Both configs start at seed 42, crop 256, batch 2, AdamW learning rate 1e-5,
weight decay 0, and Sinkhorn regularization 10 for 100 iterations. CARPK
allows up to 150 epochs, validates on epoch 1 and then every 5 epochs, and
stops after 8 validation checks without improvement. PUCPR+ allows up to 200
epochs, validates on epoch 1 and then every 2 epochs, and stops after 15
checks without improvement. Full-image patch validation is much more costly
than training-crop inference, particularly for CARPK. These are
**vehicle adaptation choices**, not the authors' 2000-epoch crowd schedule.
Batch 2 is the smallest valid batch for upstream ODConv's BatchNorm; keep
`drop_last=True`. The 256 crop matches upstream's `custom` setting, is
divisible by 32, and fits more easily on a 6 GB GPU. Increase crop and batch
only after checking memory and preserving one common protocol across models.
No scheduler is active in the authors' optimizer loop, so none is added here.
In a 1000-crop random audit at size 256, 24.5% of CARPK and 29.4% of PUCPR+
training crops had no box center. Such crops still train the count head
against zero and provide background examples. The other crops activate OT
and TV; inspect the logged components during a full run. The upstream OT
implementation can report a near-zero surrogate scalar while its gradient
remains nonzero.

## Run

From the project root, first run checks and an offline, two-image GPU smoke:

```powershell
python -m unittest discover -s tests -v
python -m vehicle_ffnet.train --config configs/carpk.json --device cuda:0 `
  --workers 0 --limit 2 --epochs 1 --no-pretrained `
  --output-dir outputs/smoke_carpk
python -m vehicle_ffnet.train --config configs/pucpr.json --device cuda:0 `
  --workers 0 --limit 2 --epochs 1 --no-pretrained `
  --output-dir outputs/smoke_pucpr
```

For full training with ImageNet initialization:

```powershell
python -u -m vehicle_ffnet.train --config configs/carpk.json --device cuda:0
python -u -m vehicle_ffnet.train --config configs/carpk_gf2_val.json --device cuda:0
python -u -m vehicle_ffnet.train --config configs/pucpr.json --device cuda:0
```

On a Linux server, keep the same relative paths when copying the complete
project, or override `--data-root` and `--manifest-dir` separately.
Each run saves `resolved_config.json`, `history.csv`, `last.pth`, and
`best_mae.pth`. If interrupted, resume using `--resume outputs/.../last.pth`
with the same output directory and configuration. The best checkpoint uses
validation MAE, not test MAE.

After selecting the settings and epoch on validation, an optional final
training run can use all official training images:

```powershell
python -u -m vehicle_ffnet.train --config configs/carpk.json `
  --train-split train_full --epochs SELECTED_EPOCH --output-dir outputs/carpk_full_seed42
```

That run has no validation and saves `last.pth`; set `SELECTED_EPOCH` to the
previously selected validation epoch, then evaluate its test checkpoint once.
For a directly comparable first baseline, evaluating `best_mae.pth` from the
827/162 run is simpler.

```powershell
python -m vehicle_ffnet.evaluate --config configs/carpk.json `
  --checkpoint outputs/carpk_ffnet_seed42/best_mae.pth `
  --split test --device cuda:0 --output-dir outputs/analysis/carpk_test
python -m vehicle_ffnet.evaluate --config configs/pucpr.json `
  --checkpoint outputs/pucpr_ffnet_seed42/best_mae.pth `
  --split test --device cuda:0 --output-dir outputs/analysis/pucpr_test
```

Evaluation writes `per_image.csv` and `summary.json` with MAE, RMSE and
signed bias. These metrics follow the fixed local data protocol, not the
paper's ShanghaiTech/NWPU comparisons. No full-run CARPK or PUCPR+ score is
claimed until the models finish training.

Local verification used an RTX 3060 Laptop GPU (6 GB): five adapter tests
passed, the official model returned a finite `[2,1,16,16]` map for a
`[2,3,128,128]` input, and both datasets completed a two-image, one-epoch
256px crop run without pretrained weights. This checks the pipeline only;
the resulting MAE is not a trained baseline. Upstream Sinkhorn emits a
PyTorch deprecation warning under torch 2.7.1, but forward/backward
completed and the OT gradient was finite and nonzero.

## Research direction after baseline

The official model already has channel attention, spatial attention and
ODConv. Adding CBAM or another generic dynamic convolution first would make
the question harder to isolate. A more targeted next change is a **light
stride-4 evidence path** from ConvNeXt's earlier feature map into the final
density head, with a residual/gate initialized near zero. This can test
whether visible car fragments missing at stride 8 recover local density
without inflating background or two-wheelers. Next, if car-size buckets on
PUCPR+/road scenes show different errors, replace fixed feature concatenation
with a small spatial scale router over the existing stride-8/16/32 branches.
Use bbox size only to supervise the router during training; no GT boxes enter
inference. Report per-vehicle local mass and background positive mass alongside
whole-image MAE. One change per experiment keeps the claim interpretable.
