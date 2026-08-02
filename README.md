# STRMSR

**Cardiac MRI Through-Plane Super-Resolution Guided by Reference and Memory**

Cardiac MRI is acquired with high in-plane resolution but thick slices, so the
through-plane direction is blurred. STRMSR recovers that missing detail by
combining two sources of guidance:

- **Reference views** — high-resolution slices from another orientation of the
  same heart, which contain the anatomy that the low-resolution view lost.
- **Memory** — previously reconstructed frames, so information is carried
  along the cardiac cycle instead of each frame being restored on its own.

## Method

The network runs three encoders: the target LR view, the LR reference and the
HR reference, each producing a 3-level feature pyramid built from Swin
Transformer groups (LR-STG / HR-STG).

| Module | Name | Role |
|---|---|---|
| CFCM | Coarse-to-Fine Contextual Matching | Matches target to reference patches coarse-to-fine, then warps reference features onto the target |
| PDFA | Patch-wise Dynamic Feature Aggregation | Pools each patch, scores it with an MLP and softmax-weights the views, so every patch picks the reference that helps it most |
| FAM | Feature Alignment Module | Aligns the warped reference features to the target features at each scale |
| DPRB | Dense Progressive Reconstruction Block | Fuses aligned features and reconstructs, coarse scale to fine |

Two memory banks supply the reference stream: past LR frames feed the
reference encoder, past reconstructions feed the HR branch. Training uses an L1
image loss plus a k-space consistency term.

## Setup

```bash
conda activate <your_env>   # Python 3.9, PyTorch >= 1.10
pip install nibabel scikit-image opencv-python tqdm pillow scipy
```

A CUDA GPU is required.

## Data

One folder per case, NIFTI volumes inside:

```
data_root/
├── GT/       <case>/*.nii.gz          ground-truth high-resolution
├── InputLR/  <case>/*.nii.gz          undersampled input, upsampled to HR size
├── RefLR/    <case>/<view>/*.nii.gz   low-resolution reference views
└── RefHR/    <case>/<view>/*.nii.gz   high-resolution reference views
```

Volumes are read as magnitude, normalized to [0,1] and treated as
pseudo-complex (real channel + zero imaginary channel).

## Training

Edit `DATA_ROOT` at the top of a script and run it:

```bash
bash options/train_4x.sh   # 4x undersampled input
bash options/train_8x.sh   # 8x undersampled input
```

Or call the entry point directly — every hyper-parameter defaults to the
cardiac 4x setup, so only `--data_root` is required:

```bash
python train.py \
  --data_root /path/to/dataset_train \
  --experiment_name STRMSR_cardiac_4x \
  --enable_temporal \
  --gpu_ids 0 1
```

Outputs land in `<output_path>/outputs/<experiment_name>/`:

```
checkpoints/model_epoch<N>.pt
train_loss.csv
metrics.csv
options.json
```

Resume with `--resume outputs/<experiment_name>/checkpoints/model_epoch<N>.pt`.

## Inference

```bash
python test.py \
  --data_root /path/to/dataset_test \
  --resume outputs/STRMSR_cardiac_4x/checkpoints/model_final.pt \
  --experiment_name STRMSR_cardiac_4x_test \
  --acceleration_factor 4 \
  --metric_mode magnitude \
  --gpu_ids 0
```

`--acceleration_factor`, `--temporal_memory`, `--height` and `--width` must
match the values used for training. Predictions are written as PNGs to
`<output_path>/outputs/<experiment_name>/Pred/<case>/`, and per-case
MSE / SSIM / PSNR are reported at the end of the run.

More usage examples: [options/stacom_cardiac.md](options/stacom_cardiac.md).

## Repository layout

```
STRMSR/
├── train.py                      training entry point
├── test.py                       batched inference + metrics
├── utils.py                      output-folder helpers
├── datasets/STRMSR_Dataset.py    NIFTI loader, temporal sequence sampling
├── models/model_STRMSR.py        training wrapper (losses, optimizer, eval)
├── networks/network_STRMSR.py    the network (CFCM, PDFA, FAM, DPRB)
└── options/                      ready-to-run training scripts
```

## Key options

| Option | Meaning |
|---|---|
| `--upscale` | Network SR factor (4). Not the undersampling rate. |
| `--acceleration_factor` | Undersampling factor of the input (4 or 8) |
| `--temporal_frames` | Frames per training sequence |
| `--temporal_memory` | Previous frames kept as memory |
| `--max_frame_gap` | Max spacing between sampled frames |
| `--enable_temporal` | Enable temporal sequence sampling |
| `--no_kspace` | Disable the k-space consistency loss |
| `--metric_mode` | `real`, `magnitude` or `both` for PSNR/SSIM |

## Citation

```bibtex
@article{pan2026cardiac,
  title={Cardiac MRI Through-Plane Super-Resolution Guided by Reference and Memory},
  author={Pan, Shaoming and Hu, Chenchuhui and Axel, Leon and Ye, Meng},
  journal={arXiv preprint arXiv:2607.07581},
  year={2026}
}
```
