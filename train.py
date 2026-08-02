#!/usr/bin/env python
"""
STRMSR training entry point
---------------------------
Spatio-Temporal Reference-based Multi-contrast MRI Super-Resolution on NIFTI data
(cardiac and brain), for 4x and 8x downsampled inputs.

Every argument below has a working default matching the cardiac 4x setup, so the
only argument you must provide is --data_root:

    python train.py \
        --data_root /path/to/dataset --enable_temporal

Expected --data_root layout (each case is a folder of *.nii.gz volumes):

    data_root/
    |-- GT/       <case>/*.nii.gz          ground-truth high-resolution
    |-- InputLR/  <case>/*.nii.gz          low-resolution input
    |-- RefLR/    <case>/<view>/*.nii.gz   low-resolution reference views
    `-- RefHR/    <case>/<view>/*.nii.gz   high-resolution reference views

Features:
- Random frame sampling per epoch
- Proper epoch resampling
- No gradient accumulation
"""

import os
import argparse
import json
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import prepare_sub_folder
from models import create_model
import scipy.io as sio  # keep (even if unused)
import csv

from datasets.STRMSR_Dataset import get_datasets_STRMSR


def parse_args():
    parser = argparse.ArgumentParser(
        description='STRMSR training on NIFTI MRI data (defaults = cardiac 4x setup)')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='STRMSR_cardiac_4x',
                        help='Name of the run; results go to <output_path>/outputs/<experiment_name>')
    parser.add_argument('--model_type', type=str, default='STRMSR',
                        help='Training wrapper in models/ (only "STRMSR" is provided)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint (.pt) to resume training from')

    # Dataset
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset root containing GT/, InputLR/, RefLR/, RefHR/ subfolders')
    parser.add_argument('--output_path', type=str, default='./',
                        help='Where to write outputs/<experiment_name>/{checkpoints,images,csv}')
    parser.add_argument('--val_cases', type=str, nargs='+', default=None,
                        help='Case names held out for validation (default: automatic split)')
    parser.add_argument('--test_cases', type=str, nargs='+', default=None,
                        help='Case names held out for testing (default: automatic split)')
    parser.add_argument('--frames_per_case', type=int, default=2,
                        help='Random single frames sampled per case per epoch (non-temporal mode)')
    parser.add_argument('--sequences_per_case', type=int, default=8,
                        help='Temporal sequences sampled per case per epoch (temporal mode)')
    parser.add_argument('--acceleration_factor', type=int, default=4,
                        help='K-space undersampling acceleration of the input (4 or 8)')

    # Temporal
    parser.add_argument('--temporal_frames', type=int, default=4,
                        help='Frames per temporal sequence (2-4); use 3 for the 8x setup')
    parser.add_argument('--max_frame_gap', type=int, default=3,
                        help='Max index gap between consecutive sampled frames (1-5)')
    parser.add_argument('--enable_temporal', action='store_true', default=False,
                        help='Enable temporal sequence sampling (recommended; off = single frame)')

    # Model
    parser.add_argument('--net_G', type=str, default='STRMSR',
                        help='Generator network in networks/ (only "STRMSR" is provided)')
    parser.add_argument('--n_recurrent', type=int, default=1,
                        help='Number of recurrent reconstruction blocks')
    parser.add_argument('--upscale', type=int, default=4,
                        help='SR factor of the network (4; the input is pre-upsampled to HR size)')
    parser.add_argument('--temporal_memory', type=int, default=2,
                        help='Number of previous frames kept as temporal memory')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Swin Transformer window size (height/width must be divisible by it)')
    parser.add_argument('--height', type=int, default=176,
                        help='Image height (176 cardiac, 144 brain)')
    parser.add_argument('--width', type=int, default=256,
                        help='Image width (256 cardiac, 240 brain)')

    # Loss
    parser.add_argument('--wr_L1', type=float, default=1.0,
                        help='Weight of the L1 reconstruction loss (0 disables image dumps)')
    parser.add_argument('--no_kspace', action='store_true',
                        help='Disable the k-space data-consistency loss (enabled by default)')

    # Training
    parser.add_argument('--n_epochs', type=int, default=250,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (2 fits a temporal sequence on a 40GB A100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Adam weight decay')

    # LR schedule
    parser.add_argument('--lr_policy', type=str, default='step',
                        help='Learning-rate schedule: step | plateau | cosine')
    parser.add_argument('--step_size', type=int, default=50,
                        help='LR decay period in epochs (step policy)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR multiplicative decay factor (step policy)')

    # Logging & Saving
    parser.add_argument('--eval_epochs', type=int, default=5,
                        help='Run validation (PSNR/SSIM) every N epochs')
    parser.add_argument('--save_epochs', type=int, default=5,
                        help='Write a checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Append training losses to train_loss.csv every N batches')

    # Hardware
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader worker processes')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1],
                        help='CUDA device ids (indices within CUDA_VISIBLE_DEVICES)')

    return parser.parse_args()


def main():
    opts = parse_args()

    # Print configuration
    options_str = json.dumps(vars(opts), indent=4, sort_keys=False)
    print("=" * 70)
    print("McMRSR Training with NIFTI Cardiac MRI - NO GRAD ACCUM")
    print("=" * 70)
    print(options_str[2:-2])
    print("=" * 70)
    print(f"\n[INFO] Batch size: {opts.batch_size}\n")

    cudnn.benchmark = True

    # Create model
    model = create_model(opts)
    model.setgpu(opts.gpu_ids)

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Parameters: {num_param:,}\n')

    # Load or initialize
    if opts.resume is None:
        model.initialize()
        start_epoch = 0
    else:
        start_epoch, _ = model.resume(opts.resume)
        start_epoch += 1

    model.set_scheduler(opts, start_epoch - 1)
    print(f'[INFO] Starting at epoch {start_epoch}\n')

    # Load datasets
    print("[INFO] Loading datasets...")
    opts.use_kspace = not opts.no_kspace
    train_set, val_set, test_set = get_datasets_STRMSR(opts)

    train_loader = DataLoader(
        train_set,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=False  # no accumulation; keep all data
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True
    )

    print(f"[INFO] Train: {len(train_set)} samples ({len(train_loader)} batches)")
    print(f"[INFO] Valid: {len(val_set)} samples")
    print(f"[INFO] Samples per epoch: {len(train_set)}\n")

    # Setup output directories
    output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    # Save config
    with open(os.path.join(output_directory, 'options.json'), 'w') as f:
        f.write(options_str)

    # Initialize loss CSV
    with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch'] + model.loss_names)

    # Initialize metrics CSV
    with open(os.path.join(output_directory, 'metrics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'psnr', 'ssim'])

    # Training loop
    print("=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(start_epoch, opts.n_epochs):
        # ========== RESAMPLE FRAMES FOR THIS EPOCH ==========
        if hasattr(train_set, 'set_epoch'):
            train_set.set_epoch(epoch)
            print(f"\n[Epoch {epoch}] Resampled {len(train_set)} training samples")

        # ========== Training ==========
        model.train()
        model.set_epoch(epoch)

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{opts.n_epochs-1}')

        for batch_idx, data in enumerate(train_bar):
            model.set_input(data)
            model.optimize()  # forward + backward + step (clipping should be inside model.update_G)

            train_bar.set_description(
                f'[Epoch {epoch}/{opts.n_epochs-1}] {model.loss_summary}'
            )

            # Log losses
            if batch_idx % opts.log_freq == 0:
                with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
                    writer = csv.writer(f)
                    losses = [epoch, batch_idx] + list(model.get_current_losses().values())
                    writer.writerow(losses)

        # Update learning rate
        model.update_learning_rate()

        # ========== Validation ==========
        if (epoch + 1) % opts.eval_epochs == 0:
            print(f"\n[Epoch {epoch}] Running validation...")

            # Save sample images
            if opts.wr_L1 > 0:
                from torchvision.utils import save_image

                pred_path = os.path.join(image_directory, f'pred_epoch{epoch:03d}.png')
                gt_path = os.path.join(image_directory, f'gt_epoch{epoch:03d}.png')
                input_path = os.path.join(image_directory, f'input_epoch{epoch:03d}.png')

                # ========== HANDLE TEMPORAL SEQUENCES ==========
                # Check if outputs are temporal [B, T, C, H, W] or single-frame [B, C, H, W]
                if model.pred_out.dim() == 5:  # Temporal
                    # Take last frame from each sequence
                    vis_pred = model.pred_out[:, -1, 0:1, :, :]     # [B, 1, H, W]
                    vis_gt = model.tag_image_full[:, -1, 0:1, :, :]  # [B, 1, H, W]
                    vis_input = model.tag_image_sub[:, -1, 0:1, :, :]  # [B, 1, H, W]
                else:  # Single-frame
                    vis_pred = model.pred_out[:, 0:1, :, :]         # [B, 1, H, W]
                    vis_gt = model.tag_image_full[:, 0:1, :, :]  # [B, 1, H, W]
                    vis_input = model.tag_image_sub[:, 0:1, :, :]  # [B, 1, H, W]

                save_image(vis_pred, pred_path, normalize=True)
                save_image(vis_gt, gt_path, normalize=True)
                save_image(vis_input, input_path, normalize=True)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                model.evaluate(val_loader)

            print(f"[Epoch {epoch}] PSNR: {model.psnr_recon:.4f}, SSIM: {model.ssim_recon:.4f}\n")

            with open(os.path.join(output_directory, 'metrics.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, model.psnr_recon, model.ssim_recon])

        # ========== Save Checkpoint ==========
        if (epoch + 1) % opts.save_epochs == 0:
            ckpt_path = os.path.join(checkpoint_directory, f'model_epoch{epoch:03d}.pt')
            model.save(ckpt_path, epoch, 0)
            print(f"[Epoch {epoch}] Saved checkpoint: {ckpt_path}\n")

    # ========== Final Save ==========
    final_ckpt = os.path.join(checkpoint_directory, 'model_final.pt')
    model.save(final_ckpt, opts.n_epochs - 1, 0)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
