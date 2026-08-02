#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py
-------------------------------------
STRMSR NIFTI inference — batched multi-volume + corrected metrics.

Processes --volume_batch_size volumes simultaneously to saturate GPU memory
and increase throughput vs one-volume-at-a-time.

Pipeline per batch:
  1. load_case_data()       – load each volume to CPU tensors
  2. run_batch_inference()  – pad to max_T, stack [B,T,...], one forward pass
  3. postprocess_case()     – save PNGs, collect tensors for metrics

Metric modes:
  real      – channel 0 only
  magnitude – sqrt(real^2 + imag^2)
  both      – report both

Usage:
  python test.py \
    --data_root /path/to/data \
    --resume /path/to/checkpoint.pt \
    --experiment_name strmsr_test \
    --volume_batch_size 4 \
    --metric_mode real \
    --gpu_ids 0 1
"""

import re
import csv
import math
import argparse
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from models import create_model
from models.utils import fft2
from models.losses import LossComputer


# ------------------------------------------------------------------ #
#  File ordering
# ------------------------------------------------------------------ #
_num_pat = re.compile(r"(\d+)")

def _extract_idx(p: Path) -> int:
    m = _num_pat.search(p.stem)
    return int(m.group(1)) if m else -1

def _list_niis_sorted(folder: Path) -> List[Path]:
    files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
    return sorted(files, key=lambda x: (_extract_idx(x) == -1, _extract_idx(x), x.name))


# ------------------------------------------------------------------ #
#  NIFTI → magnitude → pseudo-complex 2ch
# ------------------------------------------------------------------ #
def _read_mag_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 2, 3) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr[..., 0].astype(np.float32)
    raise RuntimeError(f"Unsupported NIFTI shape {arr.shape} for {path}")

def _mag_to_2ch(mag: np.ndarray) -> torch.Tensor:
    real = mag.astype(np.float32)
    return torch.from_numpy(np.stack([real, np.zeros_like(real)], axis=0)).float()  # [2,H,W]


# ------------------------------------------------------------------ #
#  PNG saving (visualisation only, no metric impact)
# ------------------------------------------------------------------ #
def _save_pred_png_ch0(recon_2ch: torch.Tensor, out_path: Path):
    img = recon_2ch[0].detach().cpu().float().numpy()
    img_u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(str(out_path))


# ------------------------------------------------------------------ #
#  Metric helpers
# ------------------------------------------------------------------ #
def _complex_to_magnitude(x2: torch.Tensor) -> torch.Tensor:
    if x2.dim() == 3:
        r, i = x2[0:1], x2[1:2]
    elif x2.dim() == 4:
        r, i = x2[:, 0:1], x2[:, 1:2]
    else:
        raise ValueError(f"Unsupported shape {tuple(x2.shape)}")
    return torch.sqrt(r * r + i * i + 1e-12)

def _to_seq_5d(x_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    frames = []
    for x in x_list:
        if x.dim() == 4:
            x = x.squeeze(0)
        frames.append(x)
    return torch.stack(frames).unsqueeze(0).to(device)  # [1, T, 1, H, W]

@torch.no_grad()
def _compute_seq_metrics(lc, pred_list, gt_list, device) -> Tuple[float, float, float]:
    pred_seq = _to_seq_5d(pred_list, device)
    gt_seq   = _to_seq_5d(gt_list,   device)
    out = lc.compute({"pred": pred_seq, "gt": gt_seq}, it=0)
    return float(out["mse"]), float(out["ssim"]), float(out["psnr"])

def _accumulate_sse(pred_list, gt_list) -> Tuple[float, int]:
    sse, n = 0.0, 0
    for p, t in zip(pred_list, gt_list):
        if p.dim() == 4: p = p.squeeze(0)
        if t.dim() == 4: t = t.squeeze(0)
        diff = p.float() - t.float()
        sse += float((diff * diff).sum())
        n   += diff.numel()
    return sse, n

def _psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    return 10.0 * math.log10(peak * peak / (mse + 1e-12))


# ------------------------------------------------------------------ #
#  Stage 1 – load one case to CPU tensors
# ------------------------------------------------------------------ #
def load_case_data(
    gt_files: List[Path],
    lr_files: List[Path],
    ref_lr_root: Path,
    ref_hr_root: Path,
    case_id: str,
    acceleration_factor: int,
    use_kspace: bool,
) -> Dict:
    """
    Load all frames + frame-specific references for one case.
    Everything stays on CPU; nothing is moved to GPU here.
    Returns a dict of tensors ready to be batched.
    """
    T = min(len(gt_files), len(lr_files))
    reflr_files = _list_niis_sorted(ref_lr_root / case_id)
    refhr_files = _list_niis_sorted(ref_hr_root / case_id)
    num_refs = len(refhr_files)

    if num_refs == 0:
        print(f"    [WARN] No refs for {case_id}, using LR/GT fallback")
    else:
        print(f"    {case_id}: T={T}, refs={num_refs}")

    gt_frames, lr_frames, k_frames = [], [], []
    ref_lr_frames, ref_hr_frames   = [], []

    for t in range(T):
        gt_2ch = _mag_to_2ch(_read_mag_nii(gt_files[t]))
        lr_2ch = _mag_to_2ch(_read_mag_nii(lr_files[t]))
        gt_frames.append(gt_2ch)
        lr_frames.append(lr_2ch)

        if use_kspace:
            k_frames.append(fft2(gt_2ch.unsqueeze(0)).squeeze(0).contiguous())

        if num_refs > 0:
            ref_base = t // acceleration_factor
            idx1 = min(ref_base,     num_refs - 1)
            idx2 = min(ref_base + 1, num_refs - 1)
            rlr, rhr = [], []
            for ridx in (idx1, idx2):
                if ridx < len(reflr_files):
                    rlr.append(_mag_to_2ch(_read_mag_nii(reflr_files[ridx])))
                if ridx < len(refhr_files):
                    rhr.append(_mag_to_2ch(_read_mag_nii(refhr_files[ridx])))
            ref_lr_frames.append(torch.stack(rlr) if rlr else lr_2ch.unsqueeze(0))
            ref_hr_frames.append(torch.stack(rhr) if rhr else gt_2ch.unsqueeze(0))
        else:
            ref_lr_frames.append(lr_2ch.unsqueeze(0))
            ref_hr_frames.append(gt_2ch.unsqueeze(0))

    return {
        'case_id':      case_id,
        'T':            T,
        'gt_files':     gt_files,
        'gt_stack':     torch.stack(gt_frames),        # [T, 2, H, W]
        'lr_stack':     torch.stack(lr_frames),        # [T, 2, H, W]
        'ref_lr_stack': torch.stack(ref_lr_frames),    # [T, Nref, 2, H, W]
        'ref_hr_stack': torch.stack(ref_hr_frames),    # [T, Nref, 2, H, W]
        'k_stack':      torch.stack(k_frames) if k_frames else None,  # [T, 2, H, W]
    }


# ------------------------------------------------------------------ #
#  Stage 2 – batched GPU inference across multiple volumes
# ------------------------------------------------------------------ #
def run_batch_inference(
    model,
    cases_data: List[Dict],
    acceleration_factor: int,   # kept for call-site compatibility; unused since the mask was removed
    device: torch.device,
    use_kspace: bool,
    do_warmup: bool = False,
    use_amp: bool = False,
) -> Tuple[List[torch.Tensor], float]:
    """
    Pad all cases to the same T, stack to [B, T, ...], run one forward pass.

    Returns:
        per_case_preds: list of [T_i, 2, H, W] CPU tensors (unpadded, one per case)
        inf_time:       wall-clock seconds for the forward pass
    """
    Ts    = [c['T'] for c in cases_data]
    max_T = max(Ts)
    B     = len(cases_data)

    # Validate Nref consistency across the batch
    Nrefs = [c['ref_hr_stack'].shape[1] for c in cases_data]
    if len(set(Nrefs)) > 1:
        raise RuntimeError(
            f"Mismatched Nref across batch: {dict(zip([c['case_id'] for c in cases_data], Nrefs))}. "
            "Reduce --volume_batch_size or ensure all cases have the same number of references."
        )

    def pad_T(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] == max_T:
            return t
        pad = torch.zeros((max_T - t.shape[0],) + t.shape[1:], dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    # Stack into batch tensors (all on CPU; model.set_input moves to device)
    gt_b    = torch.stack([pad_T(c['gt_stack'])     for c in cases_data])   # [B, T, 2, H, W]
    lr_b    = torch.stack([pad_T(c['lr_stack'])     for c in cases_data])   # [B, T, 2, H, W]
    refhr_b = torch.stack([pad_T(c['ref_hr_stack']) for c in cases_data])   # [B, T, Nref, 2, H, W]
    reflr_b = torch.stack([pad_T(c['ref_lr_stack']) for c in cases_data])   # [B, T, Nref, 2, H, W]

    data = {
        "ref_image_full": refhr_b,
        "ref_image_sub":  reflr_b,
        "tag_image_full": gt_b,
        "tag_image_sub":  lr_b,
    }

    if use_kspace:
        k_b = torch.stack([pad_T(c['k_stack']) for c in cases_data])        # [B, T, 2, H, W]
        data["tag_kspace_full"] = k_b

    # One-time GPU warmup (first batch only, avoids polluting timing)
    if do_warmup and device.type == 'cuda':
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                model.set_input(data)
                model.forward()
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            model.set_input(data)
            model.forward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    inf_time = time.time() - start

    if device.type == 'cuda':
        used_gb  = torch.cuda.memory_reserved(device) / 1e9
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  GPU mem: {used_gb:.1f}/{total_gb:.1f} GB  |  B={B}, T={max_T}")

    recon = model.pred_out  # [B, T, 2, H, W]
    if recon.dim() != 5:
        raise RuntimeError(f"Expected [B,T,2,H,W], got {tuple(recon.shape)}")

    # Unpad: each case gets its real T frames only
    per_case_preds = [recon[i, :Ts[i]].detach().cpu() for i in range(B)]
    return per_case_preds, inf_time


# ------------------------------------------------------------------ #
#  Stage 3 – save PNGs + collect tensors for metric computation
# ------------------------------------------------------------------ #
def postprocess_case(
    pred_frames: torch.Tensor,   # [T, 2, H, W] CPU
    case_data: Dict,
    pred_root: Path,
    skip_black: bool,
    black_thresh: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    T        = case_data['T']
    gt_stack = case_data['gt_stack']   # [T, 2, H, W] CPU
    gt_files = case_data['gt_files']
    case_id  = case_data['case_id']

    pred_list, gt_list, num_used = [], [], 0
    for t in range(T):
        recon_2ch = pred_frames[t]   # [2, H, W]
        gt_2ch    = gt_stack[t]      # [2, H, W]

        _save_pred_png_ch0(recon_2ch, pred_root / case_id / f"{gt_files[t].stem}.png")

        if skip_black and float(gt_2ch[0].mean()) <= black_thresh:
            continue

        pred_list.append(recon_2ch)
        gt_list.append(gt_2ch)
        num_used += 1

    return pred_list, gt_list, num_used


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def parse_args():
    ap = argparse.ArgumentParser(
        description="STRMSR batched NIFTI inference (defaults = cardiac 4x setup)")

    ap.add_argument("--data_root",       type=str, required=True,
                    help="Dataset root containing GT/, InputLR/, RefLR/, RefHR/ subfolders")
    ap.add_argument("--resume",          type=str, required=True,
                    help="Trained checkpoint (.pt) to load")
    ap.add_argument("--experiment_name", type=str, default="strmsr_test",
                    help="Run name; predictions go to <output_path>/outputs/<experiment_name>/Pred/")
    ap.add_argument("--output_path",     type=str, default="./",
                    help="Where to write predictions and metric CSVs")
    ap.add_argument("--test_cases",      type=str, nargs="+", default=None,
                    help="Case names to evaluate (default: every case under data_root)")

    # Model config — must match training
    ap.add_argument("--model_type",   type=str,   default="STRMSR",
                    help="Training wrapper in models/ (only \"STRMSR\" is provided)")
    ap.add_argument("--net_G",        type=str,   default="STRMSR",
                    help="Generator network in networks/ (only \"STRMSR\" is provided)")
    ap.add_argument("--n_recurrent",  type=int,   default=1,
                    help="Number of recurrent reconstruction blocks")
    ap.add_argument("--upscale",      type=int,   default=4,
                    help="SR factor of the network (4; the input is pre-upsampled to HR size)")
    ap.add_argument("--window_size",  type=int,   default=8,
                    help="Swin Transformer window size (height/width must be divisible by it)")
    ap.add_argument("--height",       type=int,   default=176,
                    help="Image height (176 cardiac, 144 brain)")
    ap.add_argument("--width",        type=int,   default=256,
                    help="Image width (256 cardiac, 240 brain)")
    ap.add_argument("--wr_L1",        type=float, default=1.0,
                    help="Weight of the L1 reconstruction loss (kept to match training config)")
    ap.add_argument("--no_kspace",    action="store_true",
                    help="Disable the k-space branch (enabled by default)")
    ap.add_argument("--acceleration_factor", type=int, default=4,
                    help="Undersampling factor; selects reference indices. Must match training.")
    ap.add_argument("--temporal_memory",     type=int, default=2,
                    help="Number of previous frames kept as temporal memory. Must match training.")

    # Batching
    ap.add_argument("--volume_batch_size", type=int, default=1,
                    help="Volumes processed simultaneously. WARNING: the model is a "
                         "recurrent loop over T frames; the GPU is already compute-saturated "
                         "at B=1 for most configs. B>1 adds proportional time. Use the "
                         "parallel launcher (--split_gpu) instead of increasing this.")
    ap.add_argument("--amp", action="store_true",
                    help="Use fp16 automatic mixed precision. ~2x speedup on tensor "
                         "cores, halves GPU memory so you can use a larger batch.")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="Number of parallel data-loading threads for prefetching the "
                         "next batch while the GPU processes the current one.")

    ap.add_argument("--gpu_ids",     type=int, nargs="+", default=[0],
                    help="CUDA device ids (indices within CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--metric_mode", type=str, default="real",
                    choices=["real", "magnitude", "both"],
                    help="PSNR/SSIM on channel 0 (real), sqrt(re^2+im^2) (magnitude), or both")
    ap.add_argument("--skip_black",          action="store_true",
                    help="Exclude near-empty frames from the metric averages")
    ap.add_argument("--black_mean_thresh",   type=float, default=1e-6,
                    help="Mean-intensity threshold used by --skip_black")
    ap.add_argument("--save_all_frames",     action="store_true",
                    help="Save a PNG for every frame instead of a sampled subset")

    return ap.parse_args()


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    opts = parse_args()
    cudnn.benchmark = True
    opts.use_kspace = not opts.no_kspace

    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        if max(opts.gpu_ids) >= nvis:
            raise RuntimeError(
                f"Invalid --gpu_ids {opts.gpu_ids}; visible devices={nvis}. "
                "Set CUDA_VISIBLE_DEVICES before launching."
            )
    device = torch.device(f"cuda:{opts.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading checkpoint: {opts.resume}")
    model = create_model(opts)
    model.setgpu(opts.gpu_ids)
    ep, it = model.resume(opts.resume, train=False)
    print(f"[INFO] Checkpoint: epoch={ep}, iter={it}")
    model.eval()

    lc = LossComputer({
        "tensor_range": 1.0, "max_val": 1.0,
        "a_mse": 1.0, "b_ssim": 1.0,
        "use_ms_ssim": True, "ssim_filter_size": 11,
        "ssim_filter_sigma": 1.5, "ssim_k1": 0.01, "ssim_k2": 0.03,
    })

    data_root  = Path(opts.data_root)
    gt_root    = data_root / "GT"
    lr_root    = data_root / "InputLR"
    reflr_root = data_root / "RefLR"
    refhr_root = data_root / "RefHR"

    cases = opts.test_cases or [d.name for d in sorted(gt_root.iterdir()) if d.is_dir()]

    out_root  = Path(opts.output_path) / "outputs" / opts.experiment_name
    pred_root = out_root / "Pred"
    out_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    case_csv    = out_root / "case_metrics.csv"
    overall_txt = out_root / "overall_summary.txt"

    if opts.metric_mode == "both":
        hdr = ["case_id", "num_frames_used", "num_frames_total",
               "inference_time_s", "fps",
               "mse_real", "ssim_real", "psnr_real",
               "mse_mag",  "ssim_mag",  "psnr_mag"]
    else:
        hdr = ["case_id", "num_frames_used", "num_frames_total",
               "inference_time_s", "fps", "mse", "ssim", "psnr"]

    with open(case_csv, "w", newline="") as f:
        csv.writer(f).writerow(hdr)

    total_frames = total_used_frames = 0
    total_inference_time = 0.0
    case_times: List[float] = []

    agg: Dict = {}
    if opts.metric_mode in ("real", "both"):
        agg["real"] = {"sse": 0.0, "n": 0, "mse": [], "ssim": [], "psnr": []}
    if opts.metric_mode in ("magnitude", "both"):
        agg["mag"]  = {"sse": 0.0, "n": 0, "mse": [], "ssim": [], "psnr": []}

    print(f"\n{'='*70}")
    print(f"STMISR BATCH INFERENCE")
    print(f"Cases: {len(cases)} | Volume batch size: {opts.volume_batch_size}")
    print(f"Metric mode: {opts.metric_mode} | AMP: {opts.amp} | Workers: {opts.num_workers}")
    print(f"{'='*70}\n")

    # ── Helper: load one case and return its data dict ─────────────────── #
    def _load_one_case(cid: str) -> Optional[Dict]:
        gt_files = _list_niis_sorted(gt_root / cid)
        lr_files = _list_niis_sorted(lr_root / cid)
        if not gt_files or not lr_files:
            return None
        return load_case_data(
            gt_files, lr_files, reflr_root, refhr_root,
            cid, opts.acceleration_factor, opts.use_kspace,
        )

    # ── Build list of batches up front (just case id lists) ───────────── #
    batch_id_groups = [
        cases[s:s + opts.volume_batch_size]
        for s in range(0, len(cases), opts.volume_batch_size)
    ]

    first_batch = True
    case_global_idx = 0

    # Async prefetch: load next batch on CPU threads while GPU runs current batch.
    with concurrent.futures.ThreadPoolExecutor(max_workers=opts.num_workers) as loader:

        # Pre-submit load jobs for the first batch
        def _submit_batch(batch_ids):
            futures = {cid: loader.submit(_load_one_case, cid) for cid in batch_ids}
            return futures

        pending = _submit_batch(batch_id_groups[0]) if batch_id_groups else {}

        for group_idx, batch_ids in enumerate(batch_id_groups):
            # Collect loaded data for the current batch
            batch_data: List[Dict] = []
            for cid in batch_ids:
                case_global_idx += 1
                cd = pending[cid].result()
                if cd is None:
                    print(f"[{case_global_idx}/{len(cases)}] {cid}: SKIP (empty GT or InputLR)")
                    continue
                T = cd['T']
                total_frames += T
                print(f"[{case_global_idx}/{len(cases)}] Loaded  {cid} ({T} frames)")
                batch_data.append(cd)

            # Submit load jobs for the NEXT batch while we run the current one on GPU
            if group_idx + 1 < len(batch_id_groups):
                pending = _submit_batch(batch_id_groups[group_idx + 1])

            if not batch_data:
                continue

            n_vol = len(batch_data)
            max_T = max(c['T'] for c in batch_data)
            total_T_batch = sum(c['T'] for c in batch_data)
            print(f"\n  → Batch inference: {n_vol} volume(s), max_T={max_T}, total_frames={total_T_batch}")

            # ── Stage 2: batched GPU forward pass ──────────────────────── #
            per_case_preds, batch_inf_time = run_batch_inference(
                model, batch_data, opts.acceleration_factor, device, opts.use_kspace,
                do_warmup=first_batch, use_amp=opts.amp,
            )
            first_batch = False

            batch_fps = total_T_batch / batch_inf_time if batch_inf_time > 0 else 0
            print(f"  Batch done: {batch_inf_time:.3f}s | {batch_fps:.2f} frames/s")

            # ── Stage 3: per-case post-processing + metrics ────────── #
            for case_data, pred_frames in zip(batch_data, per_case_preds):
                cid = case_data['case_id']
                T   = case_data['T']

                # Attribute wall time proportionally by frame count so per-case
                # CSV timing is meaningful even when cases have different lengths.
                case_inf_time = batch_inf_time * T / total_T_batch

                pred_list_2ch, gt_list_2ch, used = postprocess_case(
                    pred_frames, case_data, pred_root,
                    opts.skip_black, opts.black_mean_thresh,
                )

                if used == 0:
                    print(f"  [{cid}] WARN: no frames after black-frame filtering")
                    continue

                total_used_frames    += used
                total_inference_time += case_inf_time
                case_times.append(case_inf_time)
                fps = T / case_inf_time if case_inf_time > 0 else 0

                row = [cid, used, T, f"{case_inf_time:.4f}", f"{fps:.2f}"]

                if opts.metric_mode in ("real", "both"):
                    pred_r = [p[0:1] for p in pred_list_2ch]
                    gt_r   = [t[0:1] for t in gt_list_2ch]
                    mse_r, ssim_r, psnr_r = _compute_seq_metrics(lc, pred_r, gt_r, device)
                    sse_r, n_r = _accumulate_sse(pred_r, gt_r)
                    agg["real"]["sse"]  += sse_r
                    agg["real"]["n"]    += n_r
                    agg["real"]["mse"].append(mse_r)
                    agg["real"]["ssim"].append(ssim_r)
                    agg["real"]["psnr"].append(psnr_r)
                    if opts.metric_mode == "real":
                        row += [mse_r, ssim_r, psnr_r]

                if opts.metric_mode in ("magnitude", "both"):
                    pred_m = [_complex_to_magnitude(p) for p in pred_list_2ch]
                    gt_m   = [_complex_to_magnitude(t) for t in gt_list_2ch]
                    mse_m, ssim_m, psnr_m = _compute_seq_metrics(lc, pred_m, gt_m, device)
                    sse_m, n_m = _accumulate_sse(pred_m, gt_m)
                    agg["mag"]["sse"]  += sse_m
                    agg["mag"]["n"]    += n_m
                    agg["mag"]["mse"].append(mse_m)
                    agg["mag"]["ssim"].append(ssim_m)
                    agg["mag"]["psnr"].append(psnr_m)
                    if opts.metric_mode == "magnitude":
                        row += [mse_m, ssim_m, psnr_m]

                if opts.metric_mode == "both":
                    row += [mse_r, ssim_r, psnr_r, mse_m, ssim_m, psnr_m]
                    print(f"  [{cid}] Real PSNR={psnr_r:.4f} SSIM={ssim_r:.4f} | "
                          f"Mag PSNR={psnr_m:.4f} SSIM={ssim_m:.4f}")
                elif opts.metric_mode == "real":
                    print(f"  [{cid}] PSNR={psnr_r:.4f} SSIM={ssim_r:.4f} MSE={mse_r:.8f}")
                else:
                    print(f"  [{cid}] PSNR={psnr_m:.4f} SSIM={ssim_m:.4f} MSE={mse_m:.8f}")

                with open(case_csv, "a", newline="") as f:
                    csv.writer(f).writerow(row)

    # ── Overall summary ───────────────────────────────────────────── #
    lines = [
        f"Total cases evaluated: {len(cases)}",
        f"Total frames (raw): {total_frames}",
        f"Total frames (used): {total_used_frames}",
        f"Volume batch size: {opts.volume_batch_size}",
        "",
    ]

    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    if case_times:
        avg_t   = total_inference_time / len(case_times)
        avg_fps = total_frames / total_inference_time if total_inference_time > 0 else 0
        timing  = [
            "TIMING:",
            f"  Total inference time : {total_inference_time:.3f}s",
            f"  Average time/case   : {avg_t:.3f}s",
            f"  Min/Max case time   : {min(case_times):.3f}s / {max(case_times):.3f}s",
            f"  Overall FPS         : {avg_fps:.2f}",
            "",
        ]
        lines += timing
        for l in timing:
            print(l)

    def report_mode(key: str, label: str) -> Tuple[float, float]:
        d = agg[key]
        g_mse  = d["sse"] / d["n"] if d["n"] > 0 else float("nan")
        g_psnr = _psnr_from_mse(g_mse) if d["n"] > 0 else float("nan")
        c_mse  = float(np.mean(d["mse"]))  if d["mse"]  else float("nan")
        c_ssim = float(np.mean(d["ssim"])) if d["ssim"] else float("nan")
        c_psnr = float(np.mean(d["psnr"])) if d["psnr"] else float("nan")
        s_mse  = float(np.std(d["mse"]))   if d["mse"]  else float("nan")
        s_ssim = float(np.std(d["ssim"]))  if d["ssim"] else float("nan")
        s_psnr = float(np.std(d["psnr"]))  if d["psnr"] else float("nan")
        block = [
            f"[{label}]",
            f"  Global MSE  : {g_mse:.10f}",
            f"  Global PSNR : {g_psnr:.6f} dB",
            f"  Case-avg MSE  : {c_mse:.10f} ± {s_mse:.10f}",
            f"  Case-avg SSIM : {c_ssim:.6f} ± {s_ssim:.6f}",
            f"  Case-avg PSNR : {c_psnr:.6f} ± {s_psnr:.6f} dB",
            "",
        ]
        lines.extend(block)
        for l in block:
            print(l)
        return g_psnr, c_ssim

    gp_r = gs_r = gp_m = gs_m = None
    if opts.metric_mode in ("real", "both"):
        gp_r, gs_r = report_mode("real", "REAL")
    if opts.metric_mode in ("magnitude", "both"):
        gp_m, gs_m = report_mode("mag", "MAGNITUDE")

    if opts.metric_mode == "both" and gp_r is not None and gp_m is not None:
        diff = [
            "[DIFF: MAG - REAL]",
            f"  ΔPSNR (global)    : {gp_m - gp_r:+.6f} dB",
            f"  ΔSSIM (case-avg)  : {gs_m - gs_r:+.6f}",
            "",
        ]
        lines.extend(diff)
        for l in diff:
            print(l)

    with open(overall_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"{'='*70}")
    print(f"[INFO] PNGs    : {pred_root}")
    print(f"[INFO] CSV     : {case_csv}")
    print(f"[INFO] Summary : {overall_txt}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
