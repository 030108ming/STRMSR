#!/bin/bash
# -----------------------------------------------------------------------------
# STRMSR - cardiac MRI training, 4x undersampled input (STACOM-style dataset)
# -----------------------------------------------------------------------------
# Run from the STRMSR/ directory:
#     bash options/train_4x.sh
#
# Set DATA_ROOT to your own dataset. It must contain four subfolders, one folder
# per case inside each, and NIFTI (*.nii.gz) volumes inside those:
#
#     $DATA_ROOT/
#     |-- GT/       <case>/*.nii.gz          ground-truth high-resolution
#     |-- InputLR/  <case>/*.nii.gz          4x undersampled input, upsampled to HR size
#     |-- RefLR/    <case>/<view>/*.nii.gz   low-resolution reference views
#     `-- RefHR/    <case>/<view>/*.nii.gz   high-resolution reference views
#
# Results are written to $OUTPUT_PATH/outputs/$EXPERIMENT_NAME/
#     checkpoints/model_epoch<N>.pt, train_loss.csv, metrics.csv, options.json
#
# Every other hyper-parameter (optimizer, LR schedule, network size, logging
# frequency) already defaults to the values used here inside the training
# script, so only the arguments below need to be passed explicitly.
# -----------------------------------------------------------------------------

# ---- paths (EDIT THESE) -----------------------------------------------------
DATA_ROOT="/path/to/your/dataset/cardiac_down_4x_upsampled_train"
OUTPUT_PATH="./"
EXPERIMENT_NAME="STRMSR_cardiac_4x"

echo "========================================"
echo "STRMSR Training start: $(date)"
echo "========================================"

python train.py \
  --data_root "$DATA_ROOT" \
  --output_path "$OUTPUT_PATH" \
  --experiment_name "$EXPERIMENT_NAME" \
  --net_G STRMSR \
  --model_type STRMSR \
  --acceleration_factor 4 \
  --upscale 4 \
  --height 176 \
  --width 256 \
  --batch_size 2 \
  --n_epochs 250 \
  --temporal_memory 2 \
  --temporal_frames 4 \
  --max_frame_gap 3 \
  --sequences_per_case 8 \
  --enable_temporal \
  --lr 1e-4 \
  --num_workers 8 \
  --gpu_ids 0 1

# To continue an interrupted run, add:
#   --resume "$OUTPUT_PATH/outputs/$EXPERIMENT_NAME/checkpoints/model_epoch044.pt"

echo "========================================"
echo "STRMSR Training end: $(date)"
echo "========================================"
