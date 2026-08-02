# STRMSR — cardiac MRI usage

All commands are run from the `STRMSR/` directory. Replace `/path/to/...`
with your own locations.

## Dataset layout

Both training and test roots use the same structure, one folder per case,
NIFTI (`*.nii.gz`) volumes inside:

```
data_root/
├── GT/       <case>/*.nii.gz          ground-truth high-resolution
├── InputLR/  <case>/*.nii.gz          undersampled input, upsampled to HR size
├── RefLR/    <case>/<view>/*.nii.gz   low-resolution reference views
└── RefHR/    <case>/<view>/*.nii.gz   high-resolution reference views
```


## Training

Edit `DATA_ROOT` at the top of the script, then launch it. Every other
hyper-parameter already defaults to the values used in the paper, so nothing
else needs to be passed.

```bash
# 4x undersampled input
bash options/train_4x.sh

# 8x undersampled input
bash options/train_8x.sh
```

For a long run, detach it and follow the log:

```bash
mkdir -p logs
nohup bash options/train_4x.sh > logs/train_4x.txt 2>&1 &
tail -f logs/train_4x.txt
```

Checkpoints and CSV logs are written to
`<output_path>/outputs/<experiment_name>/`:

```
outputs/<experiment_name>/
├── checkpoints/model_epoch<N>.pt
├── train_loss.csv
├── metrics.csv
└── options.json
```

To continue an interrupted run, add `--resume` inside the script:

```bash
--resume "outputs/<experiment_name>/checkpoints/model_epoch044.pt"
```

## Inference

Point `--resume` at a trained checkpoint and `--data_root` at the test split.
The model-config flags must match the ones used for training.

```bash
python test.py \
  --data_root "/path/to/your/dataset/cardiac_down_4x_upsampled_test" \
  --resume "outputs/STRMSR_cardiac_4x/checkpoints/model_final.pt" \
  --output_path "./" \
  --experiment_name "STRMSR_cardiac_4x_test" \
  --model_type STRMSR \
  --net_G STRMSR \
  --acceleration_factor 4 \
  --temporal_memory 2 \
  --height 176 \
  --width 256 \
  --volume_batch_size 4 \
  --metric_mode magnitude \
  --gpu_ids 0
```

For an 8x model, change `--acceleration_factor` to `8` and point
`--data_root` / `--resume` at the corresponding 8x dataset and checkpoint.

Predictions are written as PNGs to
`<output_path>/outputs/<experiment_name>/Pred/<case>/`, and per-case
MSE / SSIM / PSNR are printed at the end of the run and saved alongside them.

### Notes

- `--acceleration_factor`, `--temporal_memory`, `--height` and `--width` must
  match the training configuration, otherwise the checkpoint will not load or
  the metrics will be meaningless.
- `--metric_mode` selects how PSNR/SSIM are computed: `real` (channel 0),
  `magnitude` (`sqrt(re² + im²)`), or `both`.
- `--volume_batch_size` processes several volumes in one forward pass. The
  network is a recurrent loop over frames and is usually already compute-bound
  at `1`, so raising it mainly costs memory.
- Add `--amp` for fp16 inference (roughly 2x faster, half the GPU memory).
