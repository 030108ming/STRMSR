#!/bin/bash
# -----------------------------------------------------------------------------
# STRMSR - cardiac MRI training, 8x undersampled input (STACOM-style dataset)
# -----------------------------------------------------------------------------
# Run from the STRMSR/ directory:
#     bash options/train_8x.sh
#
# Same dataset layout as the 4x script; only InputLR/ differs (8x undersampled).
#
#     $DATA_ROOT/
#     |-- GT/       <case>/*.nii.gz          ground-truth high-resolution
#     |-- InputLR/  <case>/*.nii.gz          8x undersampled input, upsampled to HR size
#     |-- RefLR/    <case>/<view>/*.nii.gz   low-resolution reference views
#     `-- RefHR/    <case>/<view>/*.nii.gz   high-resolution reference views
#
# Differences from the 4x recipe: --acceleration_factor 8, and shorter temporal
# sequences (3 frames, 6 sequences per case) because the harder 8x task is more
# memory hungry. --upscale stays 4: it is the network's SR factor, not the
# undersampling rate.
# -----------------------------------------------------------------------------

# ---- paths (EDIT THESE) -----------------------------------------------------
DATA_ROOT="/path/to/your/dataset/cardiac_down_8x_upsampled_train"
OUTPUT_PATH="./"
EXPERIMENT_NAME="STRMSR_cardiac_8x"

echo "========================================"
echo "STRMSR Training start: $(date)"
echo "========================================"

python train.py \
  --data_root "$DATA_ROOT" \
  --output_path "$OUTPUT_PATH" \
  --experiment_name "$EXPERIMENT_NAME" \
  --net_G STRMSR \
  --model_type STRMSR \
  --acceleration_factor 8 \
  --upscale 4 \
  --height 176 \
  --width 256 \
  --batch_size 2 \
  --n_epochs 250 \
  --temporal_memory 2 \
  --temporal_frames 3 \
  --max_frame_gap 3 \
  --sequences_per_case 6 \
  --enable_temporal \
  --lr 1e-4 \
  --num_workers 8 \
  --gpu_ids 0 1

# To continue an interrupted run, add:
#   --resume "$OUTPUT_PATH/outputs/$EXPERIMENT_NAME/checkpoints/model_epoch194.pt"

echo "========================================"
echo "STRMSR Training end: $(date)"
echo "========================================"
