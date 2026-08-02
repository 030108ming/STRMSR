# # -*- coding: utf-8 -*-
# """
# model/losses.py
# ---------------
# Differentiable MSE + (1 - SSIM / MS-SSIM) loss for MANSR.

# Key features:
# - Fully differentiable implementation in pure PyTorch (no NumPy / SciPy).
# - SSIM / MS-SSIM is computed using conv2d and avg_pool2d and participates in backprop.
# - No clamping of pred/gt inside the loss. The caller (network) should control the range
#   (e.g., by using sigmoid to keep outputs in [0,1]).
# - PSNR is derived from MSE and used for logging only (detached).
# - Interface is backward compatible with the old LossComputer:
#     * para["a_mse"] / para["mse_weight"]       : MSE weight
#     * para["b_ssim"] / para["ssim_weight"]     : SSIM weight
#     * para["tensor_range"] / para["max_val"]   : dynamic range of tensors
#     * para["use_ms_ssim"]                      : use MS-SSIM if True, single-scale SSIM if False
#     * para["ssim"] (dict) or flat ssim_* keys  : SSIM hyper-parameters
# """

# import torch
# import torch.nn.functional as F


# # =============================================================================
# # Gaussian window + (MS-)SSIM in pure PyTorch (differentiable)
# # =============================================================================

# def _gaussian_kernel(window_size: int,
#                      sigma: float,
#                      device: torch.device,
#                      dtype: torch.dtype) -> torch.Tensor:
#     """
#     Build a 2D Gaussian kernel (window_size x window_size).

#     Args:
#         window_size: Size of the Gaussian window (odd is recommended).
#         sigma: Standard deviation of the Gaussian.
#         device: Target device for the kernel tensor.
#         dtype: Target dtype for the kernel tensor.

#     Returns:
#         Tensor of shape [H, W] representing the 2D Gaussian kernel.
#     """
#     coords = torch.arange(window_size, device=device, dtype=dtype)
#     coords = coords - (window_size - 1) / 2.0  # center at 0
#     g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
#     g = g / g.sum()  # normalize 1D kernel
#     kernel_2d = g[:, None] * g[None, :]  # outer product -> 2D kernel
#     return kernel_2d  # [window_size, window_size]


# def _create_window(window_size: int,
#                    channel: int,
#                    sigma: float,
#                    device: torch.device,
#                    dtype: torch.dtype) -> torch.Tensor:
#     """
#     Create a [C, 1, window_size, window_size] Gaussian window for grouped conv2d.

#     Each channel gets the same spatial kernel but in separate groups.

#     Args:
#         window_size: Size of the Gaussian window.
#         channel: Number of image channels.
#         sigma: Standard deviation of the Gaussian.
#         device: Device for the resulting tensor.
#         dtype: Data type for the resulting tensor.

#     Returns:
#         window: Tensor of shape [channel, 1, window_size, window_size].
#     """
#     kernel_2d = _gaussian_kernel(window_size, sigma, device, dtype)
#     window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
#     return window


# def _ssim_torch(x: torch.Tensor,
#                 y: torch.Tensor,
#                 window_size: int = 11,
#                 sigma: float = 1.5,
#                 max_val: float = 1.0,
#                 k1: float = 0.01,
#                 k2: float = 0.03):
#     """
#     Single-scale SSIM (Structural Similarity) in pure PyTorch.

#     This function is differentiable and can be used inside a loss function.

#     Args:
#         x: Predicted image tensor, shape [N, C, H, W].
#         y: Ground-truth image tensor, shape [N, C, H, W].
#         window_size: Size of the Gaussian window for local statistics.
#         sigma: Standard deviation of the Gaussian window.
#         max_val: Dynamic range of the images (e.g., 1.0 for [0,1], 255.0 for [0,255]).
#         k1, k2: SSIM constants.

#     Returns:
#         ssim_per: Tensor of shape [N], SSIM value per sample.
#         cs_per:   Tensor of shape [N], contrast-structure term per sample.
#     """
#     device, dtype = x.device, x.dtype
#     C = x.shape[1]

#     window = _create_window(window_size, C, sigma, device, dtype)

#     mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
#     mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)

#     mu_x2 = mu_x * mu_x
#     mu_y2 = mu_y * mu_y
#     mu_xy = mu_x * mu_y

#     sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x2
#     sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y2
#     sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_xy

#     c1 = (k1 * max_val) ** 2
#     c2 = (k2 * max_val) ** 2

#     v1 = 2.0 * sigma_xy + c2
#     v2 = sigma_x2 + sigma_y2 + c2

#     ssim_map = ((2.0 * mu_xy + c1) * v1) / ((mu_x2 + mu_y2 + c1) * v2)
#     cs_map   = v1 / v2

#     # Average over channel and spatial dimensions to get per-sample SSIM
#     ssim_per = ssim_map.mean(dim=(1, 2, 3))
#     cs_per   = cs_map.mean(dim=(1, 2, 3))
#     return ssim_per, cs_per


# def _ms_ssim_torch(x: torch.Tensor,
#                    y: torch.Tensor,
#                    window_size: int = 11,
#                    sigma: float = 1.5,
#                    max_val: float = 1.0,
#                    k1: float = 0.01,
#                    k2: float = 0.03,
#                    weights=None) -> torch.Tensor:
#     """
#     Multi-scale SSIM (MS-SSIM) in pure PyTorch (differentiable).

#     The implementation follows the standard MS-SSIM definition:
#         MS-SSIM = (prod_{i=1..L-1} CS_i^{w_i}) * SSIM_L^{w_L}

#     Args:
#         x: Predicted image tensor, shape [N, C, H, W].
#         y: Ground-truth image tensor, shape [N, C, H, W].
#         window_size: Gaussian window size.
#         sigma: Gaussian sigma.
#         max_val: Dynamic range of the images.
#         k1, k2: SSIM constants.
#         weights: List/tuple of scale weights (length = number of scales).
#                  If None, uses the default MS-SSIM weights.

#     Returns:
#         ms_ssim: Tensor of shape [N], MS-SSIM value per sample.
#     """
#     if weights is None:
#         # Standard MS-SSIM weights (5 scales)
#         weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
#     weights = x.new_tensor(weights)
#     levels = int(weights.numel())

#     mssim_list = []
#     mcs_list   = []

#     cur_x, cur_y = x, y
#     for level in range(levels):
#         ssim_per, cs_per = _ssim_torch(
#             cur_x, cur_y,
#             window_size=window_size,
#             sigma=sigma,
#             max_val=max_val,
#             k1=k1,
#             k2=k2,
#         )
#         mssim_list.append(ssim_per)
#         mcs_list.append(cs_per)

#         # For all but the last scale, downsample by a factor of 2
#         if level < levels - 1:
#             cur_x = F.avg_pool2d(cur_x, kernel_size=2, stride=2)
#             cur_y = F.avg_pool2d(cur_y, kernel_size=2, stride=2)

#     mssim = torch.stack(mssim_list, dim=0)  # [L, N]
#     mcs   = torch.stack(mcs_list, dim=0)    # [L, N]

#     # Combine across scales: (prod_i cs_i^{w_i}) * ssim_L^{w_L}
#     cs_powers = mcs[:-1] ** weights[:-1].unsqueeze(1)  # [L-1, N]
#     ssim_power = mssim[-1] ** weights[-1]              # [N]
#     ms_ssim = torch.prod(cs_powers, dim=0) * ssim_power
#     return ms_ssim  # [N]


# # =============================================================================
# # LossComputer: MSE + (1-SSIM)  (all inside the computation graph, no clamp)
# # =============================================================================

# class LossComputer:
#     """
#     Combined loss for MANSR:

#         total_loss = a_mse * MSE + b_ssim * (1 - SSIM)

#     or, if "ssim_loss_form" == "direct":

#         total_loss = a_mse * MSE + b_ssim * SSIM

#     Notes:
#     - No clamping is applied to pred/gt inside this class.
#       It is assumed that the network output is already in the desired range
#       (e.g., via sigmoid to keep values in [0,1]).
#     - SSIM / MS-SSIM is implemented in pure torch and participates in backprop.
#     - PSNR is derived from MSE and is returned only for logging (detached).
#     """

#     def __init__(self, para):
#         super().__init__()

#         # Numerical range configuration
#         # If your images are in [0, 1], keep tensor_range = 1.0.
#         # If in [0, 255], set tensor_range = 255.0.
#         self.tensor_range = float(para.get("tensor_range", 1.0))

#         # Loss weights (keeping backward compatibility with old keys)
#         self.mse_weight  = float(para.get("a_mse", para.get("mse_weight", 1.0)))
#         self.ssim_weight = float(para.get("b_ssim", para.get("ssim_weight", 1.0)))

#         # SSIM loss form:
#         #   "one_minus" (default) -> use 1 - SSIM as a penalty term
#         #   "direct"              -> directly add SSIM to the loss (rarely used)
#         self.ssim_loss_form = str(para.get("ssim_loss_form", "one_minus")).lower()

#         # SSIM options
#         # You can use either:
#         #   - flat keys: "ssim_filter_size", "ssim_filter_sigma", "ssim_k1", "ssim_k2"
#         #   - or a nested dict: para["ssim"] = {...}
#         ssim_cfg = para.get("ssim", {})
#         if not isinstance(ssim_cfg, dict):
#             ssim_cfg = {}

#         # Fallback to tensor_range if max_val is not explicitly provided
#         self.max_val = float(para.get("max_val", para.get("tensor_range", 1.0)))

#         # Use MS-SSIM (True) or single-scale SSIM (False)
#         self.use_ms = bool(para.get("use_ms_ssim", True))

#         self.filter_size  = int(para.get("ssim_filter_size",
#                                  ssim_cfg.get("filter_size", 11)))
#         self.filter_sigma = float(para.get("ssim_filter_sigma",
#                                   ssim_cfg.get("filter_sigma", 1.5)))
#         self.k1           = float(para.get("ssim_k1", ssim_cfg.get("k1", 0.01)))
#         self.k2           = float(para.get("ssim_k2", ssim_cfg.get("k2", 0.03)))
#         self.weights      = para.get("ssim_weights", None)

#         # Stores the latest loss values (detached) for external inspection
#         self.last_losses = {}

#     # ----------------- helper methods -----------------

#     @staticmethod
#     def _flatten_spatiotemporal(x: torch.Tensor):
#         """
#         Flatten a spatiotemporal tensor from (B, T, C, H, W) to (B*T, C, H, W).

#         This allows SSIM / MS-SSIM to be computed across all frames in one pass.

#         If the input is already 4D (B, C, H, W), it is returned as-is.

#         Args:
#             x: Input tensor of shape (B,T,C,H,W) or (B,C,H,W).

#         Returns:
#             x_flat: 4D tensor of shape (B*T, C, H, W) or original (B,C,H,W).
#             shape_info: (B, T) if 5D, or (B, 1) if 4D.
#         """
#         if x.dim() == 5:
#             B, T, C, H, W = x.shape
#             return x.reshape(B * T, C, H, W), (B, T)
#         if x.dim() == 4:
#             return x, (x.shape[0], 1)
#         raise ValueError(f"Unsupported tensor dim {x.dim()} (expected 4D or 5D).")

#     @staticmethod
#     def _psnr_from_mse(mse: torch.Tensor,
#                        peak: float = 1.0,
#                        eps: float = 1e-10) -> torch.Tensor:
#         """
#         Compute PSNR from MSE:

#             PSNR = 10 * log10(peak^2 / MSE)

#         For peak=1, this simplifies to:

#             PSNR = -10 * log10(MSE)

#         Args:
#             mse: Scalar tensor of mean-squared error.
#             peak: Max value of the signal (e.g., 1.0 or 255.0).
#             eps: Small epsilon to avoid log(0).

#         Returns:
#             psnr: Scalar tensor with PSNR in dB.
#         """
#         return 10.0 * torch.log10((peak * peak) / (mse + eps))

#     # ----------------- main compute method -----------------

#     def compute(self, data, it):
#         """
#         Compute the combined MSE + (1-SSIM) loss.

#         Args:
#             data: dict containing:
#                 - "pred": (B,T,C,H,W) or (B,C,H,W) network outputs.
#                 - "gt"  : same shape as "pred", ground-truth images.
#                   (or key "target" for backward compatibility)
#             it: Current training iteration (not explicitly used here, kept for API).

#         Returns:
#             losses: dict with keys:
#                 - "mse":       detached MSE scalar
#                 - "mse_loss":  same as "mse"
#                 - "ssim":      detached SSIM (or MS-SSIM) scalar
#                 - "ssim_loss": detached SSIM penalty (1 - SSIM or SSIM itself)
#                 - "psnr":      detached PSNR scalar (dB)
#                 - "total_loss": scalar tensor with gradients, to be used for backward()
#         """
#         pred   = data.get("pred")
#         target = data.get("gt", data.get("target"))

#         if pred is None or target is None:
#             raise KeyError("Loss computation requires 'pred' and 'gt' (or 'target') tensors.")
#         if pred.shape != target.shape:
#             raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

#         target = target.to(pred.device, dtype=pred.dtype)

#         # ----- MSE -----
#         pred_bt, _   = self._flatten_spatiotemporal(pred)
#         target_bt, _ = self._flatten_spatiotemporal(target)

#         mse_val = F.mse_loss(pred_bt, target_bt, reduction="mean").to(pred.dtype)
#         psnr_val = self._psnr_from_mse(mse_val, peak=self.tensor_range).to(pred.dtype)

#         # ----- (MS-)SSIM (fully differentiable) -----
#         if self.use_ms:
#             ssim_per = _ms_ssim_torch(
#                 pred_bt,
#                 target_bt,
#                 window_size=self.filter_size,
#                 sigma=self.filter_sigma,
#                 max_val=self.max_val,
#                 k1=self.k1,
#                 k2=self.k2,
#                 weights=self.weights,
#             )
#         else:
#             ssim_per, _ = _ssim_torch(
#                 pred_bt,
#                 target_bt,
#                 window_size=self.filter_size,
#                 sigma=self.filter_sigma,
#                 max_val=self.max_val,
#                 k1=self.k1,
#                 k2=self.k2,
#             )

#         # Average across all samples (and frames) to get a scalar SSIM
#         ssim_tensor = ssim_per.mean()

#         # Convert SSIM to a penalty term based on the configured form
#         if self.ssim_loss_form == "direct":
#             ssim_part = ssim_tensor
#         else:
#             # Default: use (1 - SSIM) so that higher SSIM => smaller penalty
#             ssim_part = 1.0 - ssim_tensor

#         # ----- Combined total loss -----
#         total_loss = self.mse_weight * mse_val + self.ssim_weight * ssim_part

#         # For logging, we detach everything except total_loss
#         losses = {
#             "mse": mse_val.detach(),
#             "mse_loss": mse_val.detach(),
#             "ssim": ssim_tensor.detach(),
#             "ssim_loss": ssim_part.detach(),
#             "psnr": psnr_val.detach(),
#             "total_loss": total_loss,  # this one is used for backward()
#         }

#         # Cache a CPU copy for external summary/logging if needed
#         self.last_losses = {
#             k: (v.detach().cpu() if torch.is_tensor(v) else v)
#             for k, v in losses.items()
#         }
#         return losses



# use the mkssims loss computer for all models for now, since it is more stable than the previous one. We can always add more loss options in the future if needed.

# -*- coding: utf-8 -*-
"""
model/losses.py
---------------
LossComputer for MANSR (0–1 scale)
- Uses scikit-image SSIM for robust, validated implementation
- All metrics computed directly in the same scale as your tensors (default 0–1).
- MSE: standard mean-squared error on [0, tensor_range].
- PSNR: uses peak = tensor_range (default 1.0).
- SSIM: scikit-image implementation (non-differentiable), validated and fast.

Notes:
- If your training/evaluation tensors are in [0,1], keep tensor_range=1.0.
- If you ever switch to [0,255] tensors, set tensor_range=255.0 and everything
  (PSNR, SSIM) will adapt automatically.
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim


# =============================================================================
# LossComputer: MSE + (1-SSIM) with PSNR logging (all on 0–1 scale by default)
# =============================================================================

class LossComputer:
    """
    total_loss = a_mse * MSE + b_ssim * (1 - SSIM)

    - All computations are done in the same scale as your tensors.
      By default, tensor_range=1.0 (i.e., images in [0,1]).
    - SSIM is computed via scikit-image for accuracy and used for logging/metrics;
      it does not backpropagate gradients.
    """

    def __init__(self, para):
        super().__init__()

        # Scale configuration
        # If your tensors are in [0,1], keep tensor_range=1.0.
        # If they are in [0,255], set tensor_range=255.0.
        self.tensor_range = float(para.get("tensor_range", 1.0))

        # Loss weights (keep backward compatibility with old keys)
        self.mse_weight   = float(para.get("a_mse", para.get("mse_weight", 1.0)))
        self.ssim_weight  = float(para.get("b_ssim", para.get("ssim_weight", 1.0)))

        # If you want to directly add SSIM (rare), set "direct";
        # default is "one_minus" which uses (1 - SSIM).
        self.ssim_loss_form = str(para.get("ssim_loss_form", "one_minus"))

        # SSIM options for scikit-image
        self.use_ms = bool(para.get("use_ms_ssim", True))  # kept for compatibility
        self.win_size = int(para.get("ssim_filter_size", 11))  # window size
        self.sigma = float(para.get("ssim_filter_sigma", 1.5))  # Gaussian sigma
        self.k1 = float(para.get("ssim_k1", 0.01))
        self.k2 = float(para.get("ssim_k2", 0.03))

        self.last_losses  = {}

    # ----------------- helpers -----------------

    def _to_numpy_hwc(self, x: torch.Tensor) -> np.ndarray:
        """
        Convert a single (C,H,W) tensor to numpy (H,W,C), clamped to [0, tensor_range].
        """
        x = x.detach().cpu().float()
        x = x.clamp(0.0, self.tensor_range)
        x = x.permute(1, 2, 0).contiguous()
        return x.numpy().astype(np.float64)  # float64 for better precision

    @staticmethod
    def _flatten_spatiotemporal(x: torch.Tensor):
        """
        Flatten (B,T,C,H,W) -> (B*T,C,H,W) to iterate frames in a batch;
        If already (B,C,H,W), it's returned as-is.
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W), (B, T)
        if x.dim() == 4:
            return x, (x.shape[0], 1)
        raise ValueError(f"Unsupported tensor dim {x.dim()} (expected 4D or 5D).")

    @staticmethod
    def _psnr_from_mse(mse: torch.Tensor, peak: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
        """
        PSNR = 10 * log10(peak^2 / MSE). For peak=1, PSNR = -10 * log10(MSE).
        """
        return 10.0 * torch.log10((peak * peak) / (mse + eps))

    # ----------------- main -----------------

    def compute(self, data, it):
        """
        Args:
            data: dict with keys:
                - "pred": (B,T,C,H,W) or (B,C,H,W) in [0, tensor_range]
                - "gt"  : same shape as "pred"
        Returns:
            dict with tensors:
                - "mse": scalar MSE
                - "ssim": scalar SSIM
                - "psnr": scalar PSNR (dB)
                - "total_loss": weighted sum of MSE and SSIM loss part
        """
        pred   = data.get("pred")
        target = data.get("gt", data.get("target"))
        if pred is None or target is None:
            raise KeyError("Loss computation requires 'pred' and 'gt' tensors.")
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        # Ensure shape/scale/dtype/placement match
        target = target.to(pred.device, dtype=pred.dtype)

        # ----- MSE (in [0, tensor_range]) -----
        pred_bt, _   = self._flatten_spatiotemporal(pred)
        target_bt, _ = self._flatten_spatiotemporal(target)
        mse_val = F.mse_loss(pred_bt, target_bt, reduction="mean").to(pred.dtype)
        psnr_val = self._psnr_from_mse(mse_val, peak=self.tensor_range).to(pred.dtype)

        # ----- SSIM with scikit-image (no grad) -----
        # We iterate frame-by-frame for consistency
        pred_np_bt, _   = self._flatten_spatiotemporal(pred.detach())
        target_np_bt, _ = self._flatten_spatiotemporal(target.detach())

        ssim_vals = []
        for p, t in zip(pred_np_bt, target_np_bt):
            p_np = self._to_numpy_hwc(p)  # (H,W,C)
            t_np = self._to_numpy_hwc(t)  # (H,W,C)
            
            # Handle grayscale vs RGB
            if p_np.shape[2] == 1:
                # Grayscale: squeeze to (H,W)
                p_np = p_np.squeeze(axis=2)
                t_np = t_np.squeeze(axis=2)
                channel_axis = None
            else:
                # RGB: keep (H,W,C) and specify channel_axis
                channel_axis = -1
            
            # Compute SSIM using scikit-image
            score = sk_ssim(
                p_np, t_np,
                data_range=self.tensor_range,
                channel_axis=channel_axis,
                gaussian_weights=True,
                sigma=self.sigma,
                use_sample_covariance=False,
                win_size=self.win_size,
                K1=self.k1,
                K2=self.k2
            )
            ssim_vals.append(float(score))
        
        mean_ssim = float(np.mean(ssim_vals)) if len(ssim_vals) > 0 else 0.0

        ssim_tensor = torch.tensor(mean_ssim, device=pred.device, dtype=pred.dtype)
        # Default "one_minus": use 1 - SSIM as a penalty term.
        ssim_part = (1.0 - ssim_tensor) if self.ssim_loss_form.lower() != "direct" else ssim_tensor

        # ----- Weighted total loss -----
        total_loss = self.mse_weight * mse_val + self.ssim_weight * ssim_part

        # ----- Package outputs (and cache for external access) -----
        losses = {
            "mse": mse_val.detach(),
            "mse_loss": mse_val.detach(),
            "ssim": ssim_tensor.detach(),
            "ssim_loss": ssim_part.detach(),
            "psnr": psnr_val.detach(),
            "total_loss": total_loss,
        }
        self.last_losses = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                            for k, v in losses.items()}
        return losses