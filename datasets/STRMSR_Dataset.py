"""
STRMSR_Dataset.py (magnitude -> pseudo-complex 2ch, LR as input)
WITH TEMPORAL SEQUENCE SUPPORT and McMRSR-style reference selection
----------------------------------------------------------------------
Reference selection logic: index//upscale and index//upscale + 1
"""

from pathlib import Path
from typing import Dict, List, Union, Tuple
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from models.utils import fft2  # [b,2,h,w] -> [b,2,h,w]


def _list_niis(folder: Path) -> List[Path]:
    """List *.nii and *.nii.gz sorted."""
    if not folder.exists():
        return []
    return sorted(list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz")))


def _read_mag_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)

    # squeeze 
    if arr.ndim == 3:
        #  [C,H,W] 
        if arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))  # -> [H,W,C]

    if arr.ndim == 2:
        mag = arr
    elif arr.ndim == 3:
        # [H,W,C]
        mag = arr[..., 0]
    else:
        raise RuntimeError(f"Unsupported NIFTI shape: {arr.shape} for {path}")

    mag = mag.astype(np.float32)
    return mag


def _mag_to_pseudo_complex_2ch(mag: np.ndarray) -> np.ndarray:
    """
    magnitude -> pseudo-complex (2ch):
    real = mag, imag = 0
    output [2,H,W] float32
    """
    if mag.ndim != 2:
        raise RuntimeError(f"Expected mag [H,W], got {mag.shape}")
    real = mag
    imag = np.zeros_like(mag, dtype=np.float32)
    out = np.stack([real, imag], axis=0).astype(np.float32)
    return out


class STRMSR_Dataset(Dataset):
    """
    STRMSR dataset with temporal sequence support and McMRSR-style reference selection.
    
    Directory structure:
    root/
      GT/<case>/*.nii.gz
      InputLR/<case>/*.nii.gz
      RefLR/<case>/*.nii.gz    (no subdirectories like McMRSR RefLR/<case>/<view>/)
      RefHR/<case>/*.nii.gz    (no subdirectories)
    
    Reference selection logic (same as McMRSR):
      ref_base = frame_idx // acceleration_factor
      ref_indices = [ref_base, ref_base + 1]
    
    Examples (acceleration=8, 160 input frames, 20 ref frames):
      frame_idx=0-7   → ref=[0, 1]
      frame_idx=8-15  → ref=[1, 2]
      frame_idx=16-23 → ref=[2, 3]
    
    Temporal mode:
    - Samples sequences of frames with gaps 1-5
    - For training: random sequences
    - For val/test: all valid sequences
    """

    def __init__(
        self,
        root: Union[str, Path],
        mode: str = "TRAIN",
        gt_dir: str = "GT",
        lr_dir: str = "InputLR",
        reflr_dir: str = "RefLR",
        refhr_dir: str = "RefHR",
        frames_per_case: int = 6,
        sequences_per_case: int = 4,
        use_kspace: bool = True,
        val_cases: List[str] = None,
        test_cases: List[str] = None,
        # Temporal parameters
        temporal_frames: int = 4,
        max_frame_gap: int = 5,
        enable_temporal: bool = True,
        acceleration_factor: int = 8,
    ):
        self.root = Path(root)
        self.mode = mode.upper()
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.reflr_dir = reflr_dir
        self.refhr_dir = refhr_dir
        self.use_kspace = use_kspace
        self.frames_per_case = frames_per_case
        self.sequences_per_case = sequences_per_case
        
        # Temporal settings
        self.temporal_frames = temporal_frames
        self.max_frame_gap = max_frame_gap
        self.enable_temporal = enable_temporal
        
        # Mask settings
        self.acceleration_factor = acceleration_factor

        # Get all cases
        gt_root = self.root / self.gt_dir
        all_cases = [d.name for d in sorted(gt_root.iterdir()) if d.is_dir()]

        # Split cases
        if self.mode == "VALI":
            if val_cases is None:
                random.seed(42)
                available = [c for c in all_cases if c not in (test_cases or [])]
                val_cases = random.sample(available, min(4, len(available)))
            self.cases = val_cases

        elif self.mode == "TEST":
            self.cases = test_cases if (test_cases is not None and len(test_cases) > 0) else all_cases

        else:  # TRAIN
            exclude = set(val_cases or []) | set(test_cases or [])
            self.cases = [c for c in all_cases if c not in exclude]

        # Count frames per case
        self.case_info = {}
        for case_id in self.cases:
            gt_files = _list_niis(self.root / self.gt_dir / case_id)
            lr_files = _list_niis(self.root / self.lr_dir / case_id)
            ref_files = _list_niis(self.root / self.refhr_dir / case_id)
            
            num_frames = min(len(gt_files), len(lr_files))
            num_refs = len(ref_files)
            
            self.case_info[case_id] = {
                'num_frames': num_frames,
                'num_refs': num_refs
            }

        # Generate samples
        self._generate_epoch_samples()

        total_frames = sum(info['num_frames'] for info in self.case_info.values())
        temporal_str = f"temporal={temporal_frames}frames, gap≤{max_frame_gap}" if enable_temporal else "single-frame"
        print(
            f"[STRMSR {self.mode}] {len(self.cases)} cases, "
            f"{total_frames} total frames -> {len(self.samples)} samples/epoch ({temporal_str}), "
            f"accel={acceleration_factor}x"
        )

    def _sample_temporal_sequence(self, case_id: str, start_frame: int = None) -> List[int]:
        """Sample a temporal sequence with gaps between 1 and max_frame_gap."""
        num_frames = self.case_info[case_id]['num_frames']
        
        if start_frame is None:
            max_start = num_frames - self.temporal_frames
            if max_start < 0:
                return [min(i, num_frames-1) for i in range(self.temporal_frames)]
            start_frame = random.randint(0, max_start)
        
        sequence = [start_frame]
        current = start_frame
        
        for _ in range(self.temporal_frames - 1):
            gap = random.randint(1, self.max_frame_gap)
            next_frame = min(current + gap, num_frames - 1)
            sequence.append(next_frame)
            current = next_frame
        
        return sequence

    def _generate_epoch_samples(self):
        """Generate samples for the epoch."""
        self.samples = []
        
        if self.mode == "TRAIN":
            for case_id in self.cases:
                num_frames = self.case_info[case_id]['num_frames']
                if num_frames <= 0:
                    continue
                
                if self.enable_temporal:
                    for _ in range(self.sequences_per_case):
                        seq = self._sample_temporal_sequence(case_id)
                        self.samples.append((case_id, seq))
                else:
                    n_sample = min(self.frames_per_case, num_frames)
                    frame_indices = random.sample(range(num_frames), n_sample)
                    for frame_idx in frame_indices:
                        self.samples.append((case_id, [frame_idx]))
        
        else:
            # VAL/TEST: all sequences in order
            for case_id in self.cases:
                num_frames = self.case_info[case_id]['num_frames']
                
                if self.enable_temporal:
                    stride = max(1, self.temporal_frames // 2)
                    for start_idx in range(0, num_frames, stride):
                        if start_idx + self.temporal_frames > num_frames:
                            break
                        seq = [min(start_idx + i, num_frames - 1) for i in range(self.temporal_frames)]
                        self.samples.append((case_id, seq))
                else:
                    for frame_idx in range(num_frames):
                        self.samples.append((case_id, [frame_idx]))

    def set_epoch(self, epoch: int):
        """Resample training data for new epoch."""
        if self.mode == "TRAIN":
            random.seed(42 + epoch)
            self._generate_epoch_samples()

    def _get_reference_indices(self, frame_idx: int, num_refs: int) -> Tuple[int, int]:
        """
        Get two reference frame indices using McMRSR logic.
        
        Logic:
            ref_base = frame_idx // acceleration_factor
            ref_indices = [ref_base, ref_base + 1]
        
        Examples (acceleration=8, 160 input frames, 20 ref frames):
            frame_idx=0-7   → ref=[0, 1]
            frame_idx=8-15  → ref=[1, 2]
            frame_idx=16-23 → ref=[2, 3]
            frame_idx=32    → ref=[4, 5]
        
        Examples (acceleration=4, 160 input frames, 40 ref frames):
            frame_idx=0-3   → ref=[0, 1]
            frame_idx=4-7   → ref=[1, 2]
            frame_idx=16    → ref=[4, 5]
        """
        ref_base = frame_idx // self.acceleration_factor
        ref_idx1 = min(ref_base, num_refs - 1)
        ref_idx2 = min(ref_base + 1, num_refs - 1)
        
        return ref_idx1, ref_idx2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, frame_indices = self.samples[idx]
        
        # Get file lists
        gt_files = _list_niis(self.root / self.gt_dir / case_id)
        lr_files = _list_niis(self.root / self.lr_dir / case_id)
        if len(gt_files) == 0 or len(lr_files) == 0:
            raise RuntimeError(f"Empty GT/LR for case {case_id}")
        
        reflr_files = _list_niis(self.root / self.reflr_dir / case_id)
        refhr_files = _list_niis(self.root / self.refhr_dir / case_id)
        num_refs = self.case_info[case_id]['num_refs']
        
        # --- Load frames with FRAME-SPECIFIC references ---
        gt_frames = []
        lr_frames = []
        k_fulls = []
        ref_lr_frames = []  # references for each frame
        ref_hr_frames = []  # references for each frame
        
        for frame_idx in frame_indices:
            # Read magnitude
            gt_mag = _read_mag_nii(gt_files[frame_idx])
            lr_mag = _read_mag_nii(lr_files[frame_idx])
            
            # Convert to pseudo-complex
            gt_2ch = _mag_to_pseudo_complex_2ch(gt_mag)
            lr_2ch = _mag_to_pseudo_complex_2ch(lr_mag)
            
            gt_frame = torch.from_numpy(gt_2ch).float()  # [2,H,W]
            lr_frame = torch.from_numpy(lr_2ch).float()  # [2,H,W]
            
            gt_frames.append(gt_frame)
            lr_frames.append(lr_frame)
            
            # K-space
            if self.use_kspace:
                k_full = fft2(gt_frame.unsqueeze(0)).squeeze(0).contiguous()
                k_fulls.append(k_full)
            
            # Load references for THIS specific frame
            ref_idx1, ref_idx2 = self._get_reference_indices(frame_idx, num_refs)
            
            ref_lr_list = []
            ref_hr_list = []
            
            # Load two reference frames for this frame_idx
            for ref_idx in [ref_idx1, ref_idx2]:
                # RefLR
                if ref_idx < len(reflr_files):
                    ref_lr_mag = _read_mag_nii(reflr_files[ref_idx])
                    ref_lr_2ch = _mag_to_pseudo_complex_2ch(ref_lr_mag)
                    ref_lr_list.append(torch.from_numpy(ref_lr_2ch).float())
                
                # RefHR
                if ref_idx < len(refhr_files):
                    ref_hr_mag = _read_mag_nii(refhr_files[ref_idx])
                    ref_hr_2ch = _mag_to_pseudo_complex_2ch(ref_hr_mag)
                    ref_hr_list.append(torch.from_numpy(ref_hr_2ch).float())
            
            # Fallback
            if len(ref_lr_list) == 0:
                ref_lr_list = [lr_frame]
            if len(ref_hr_list) == 0:
                ref_hr_list = [gt_frame]
            
            # Stack references for this frame: [Nref,2,H,W]
            ref_lr_stack = torch.stack(ref_lr_list, dim=0)
            ref_hr_stack = torch.stack(ref_hr_list, dim=0)
            
            ref_lr_frames.append(ref_lr_stack)
            ref_hr_frames.append(ref_hr_stack)
        
        # --- Stack all frames ---
        if len(frame_indices) == 1:
            # Single frame mode: [2,H,W]
            gt_stack = gt_frames[0]
            lr_stack = lr_frames[0]
            k_stack = k_fulls[0] if self.use_kspace else None
            ref_lr_final = ref_lr_frames[0]  # [Nref,2,H,W]
            ref_hr_final = ref_hr_frames[0]  # [Nref,2,H,W]
        else:
            # Temporal sequence mode: [T,2,H,W]
            gt_stack = torch.stack(gt_frames, dim=0)
            lr_stack = torch.stack(lr_frames, dim=0)
            k_stack = torch.stack(k_fulls, dim=0) if self.use_kspace else None
            
            # Stack references: [T,Nref,2,H,W]
            ref_lr_final = torch.stack(ref_lr_frames, dim=0)  # [T,Nref,2,H,W]
            ref_hr_final = torch.stack(ref_hr_frames, dim=0)  # [T,Nref,2,H,W]
        
        # --- Output ---
        sample = {
            "ref_image_full": ref_hr_final,     # [Nref,2,H,W] or [T,Nref,2,H,W]
            "ref_image_sub":  ref_lr_final,     # [Nref,2,H,W] or [T,Nref,2,H,W]
            "tag_image_full": gt_stack,         # [2,H,W] or [T,2,H,W]
            "tag_image_sub":  lr_stack,         # [2,H,W] or [T,2,H,W]
        }
        
        if self.use_kspace:
            sample["tag_kspace_full"] = k_stack      # [2,H,W] or [T,2,H,W]

        return sample


def get_datasets_STRMSR(opts):
    """
    Factory function to create train/val/test datasets.
    
    Temporal args:
      opts.temporal_frames (int): Number of frames per sequence (default: 4)
      opts.max_frame_gap (int): Max gap between frames (default: 5)
      opts.enable_temporal (bool): Enable temporal mode (default: True)
      opts.acceleration_factor (int): Mask undersampling factor, e.g. 4 or 8 (default: 8)
    """
    use_kspace = getattr(opts, "use_kspace", True)
    val_cases = getattr(opts, "val_cases", None)
    test_cases = getattr(opts, "test_cases", None)
    frames_per_case = getattr(opts, "frames_per_case", 16)
    sequences_per_case = getattr(opts, "sequences_per_case", 4)
    
    # Temporal settings
    temporal_frames = getattr(opts, "temporal_frames", 4)
    max_frame_gap = getattr(opts, "max_frame_gap", 5)
    enable_temporal = getattr(opts, "enable_temporal", True)
    
    # Mask settings
    acceleration_factor = getattr(opts, "acceleration_factor", 8)
    
    # Auto-select val cases
    if val_cases is None:
        gt_root = Path(opts.data_root) / "GT"
        all_cases = [d.name for d in sorted(gt_root.iterdir()) if d.is_dir()]
        random.seed(42)
        available = [c for c in all_cases if c not in (test_cases or [])]
        val_cases = random.sample(available, min(4, len(available)))
        print(f"[INFO] Val cases: {val_cases}")
    
    common_kwargs = dict(
        use_kspace=use_kspace,
        frames_per_case=frames_per_case,
        val_cases=val_cases,
        test_cases=test_cases,
        temporal_frames=temporal_frames,
        max_frame_gap=max_frame_gap,
        enable_temporal=enable_temporal,
        acceleration_factor=acceleration_factor,
    )
    
    train_set = STRMSR_Dataset(
        root=opts.data_root,
        mode="TRAIN",
        sequences_per_case=sequences_per_case,
        **common_kwargs,
    )
    
    val_set = STRMSR_Dataset(
        root=opts.data_root,
        mode="VALI",
        **common_kwargs,
    )
    
    test_set = STRMSR_Dataset(
        root=opts.data_root,
        mode="TEST",
        **common_kwargs,
    )
    
    return train_set, val_set, test_set


# ============================================
# Test script
# ============================================
if __name__ == "__main__":
    class MockOpts:
        data_root = "/path/to/your/dataset"
        use_kspace = True
        val_cases = None
        test_cases = None
        frames_per_case = 16
        sequences_per_case = 4
        temporal_frames = 4
        max_frame_gap = 5
        enable_temporal = True
        acceleration_factor = 8
    
    opts = MockOpts()
    train_set, val_set, test_set = get_datasets_STRMSR(opts)
    
    print("\n=== Reference Frame Selection Examples ===")
    print("Acceleration factor: 8")
    for frame_idx in [0, 7, 8, 15, 16, 32, 64, 159]:
        ref_idx1, ref_idx2 = train_set._get_reference_indices(frame_idx, 20)
        print(f"  Frame {frame_idx:3d} → Ref frames [{ref_idx1}, {ref_idx2}]")
    
    print("\n=== Testing Train Set ===")
    if len(train_set) > 0:
        sample = train_set[0]
        print(f"\nSample 0:")
        print(f"  ref_image_full shape: {sample['ref_image_full'].shape}")
        print(f"  ref_image_sub shape: {sample['ref_image_sub'].shape}")
        print(f"  tag_image_full shape: {sample['tag_image_full'].shape}")
        print(f"  tag_image_sub shape: {sample['tag_image_sub'].shape}")
        if 'tag_kspace_full' in sample:
            print(f"  tag_kspace_full shape: {sample['tag_kspace_full'].shape}")