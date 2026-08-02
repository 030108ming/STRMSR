import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log10
from torch.optim import lr_scheduler
import scipy.io as sio
import pdb


def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()


class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_scheduler(optimizer, opts, last_epoch=-1):
    if 'lr_policy' not in opts or opts.lr_policy == 'constant':
        scheduler = None
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.step_size,
                                        gamma=opts.gamma, last_epoch=last_epoch)
    elif opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.epoch_decay) / float(opts.n_epochs - opts.epoch_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opts.lr_policy)
    return scheduler


def get_recon_loss(opts):
    loss = None
    if opts['recon'] == 'L2':
        loss = nn.MSELoss()
    elif opts['recon'] == 'L1':
        loss = nn.L1Loss()

    return loss


# def psnr(sr_image, gt_image):
#     assert sr_image.size(0) == gt_image.size(0) == 1

#     peak_signal = (gt_image.max() - gt_image.min()).item()

#     mse = (sr_image - gt_image).pow(2).mean().item()

#     return 10 * log10(peak_signal ** 2 / mse)

def psnr(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    peak_signal = 1.0  # Fixed peak for normalized [0,1] images

    mse = (sr_image - gt_image).pow(2).mean().item()
    
    if mse == 0:
        return float('inf')  # Perfect match
    
    if mse < 1e-10:  # Handle very small MSE
        mse = 1e-10

    return 10 * log10(peak_signal ** 2 / mse)

def mse(sr_image, gt_image):
    assert sr_image.size(0) == gt_image.size(0) == 1

    mse = (sr_image - gt_image).pow(2).mean().item()

    return mse


'''
K-Space
'''
# def data_consistency(k, k0, mask, noise_lvl=None):
#     """
#     k    - input in k-space [b,w,h,2] need to [b,2,w,h]
#     k0   - initially sampled elements in k-space
#     dc_mask - corresponding nonzero location
#     """
#     # k = k.permute(0,3,1,2)   #[b,2,w,h] do not permute here
#     v = noise_lvl
#     if v:  # noisy case
#         out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
#     else:  # noiseless case
#         out = (1 - mask) * k + mask * k0
#     return out


# class DataConsistencyInKspace_I(nn.Module):
#     """ Create data consistency operator

#     Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
#     This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
#     and applies FFT2 to the (nx, ny) axis.

#     """

#     def __init__(self, noise_lvl=None):
#         super(DataConsistencyInKspace_I, self).__init__()
#         self.noise_lvl = noise_lvl

#     def forward(self, *input, **kwargs):
#         return self.perform(*input)

#     # def perform(self, x, k0, mask):
#     #     """
#     #     x    - input in image domain, of shape (n, 2, nx, ny)
#     #     k0   - initially sampled elements in k-space
#     #     dc_mask - corresponding nonzero location
#     #     """

#     #     if x.dim() == 4: # input is 2D
#     #         x = x.permute(0, 2, 3, 1) #[n,w,h,2]
#     #     else:
#     #         raise ValueError("error in data consistency layer!")

#     #     k = fft2(x)
#     #     # out = data_consistency(k, k0, dc_mask.repeat(1, 1, 1, 2), self.noise_lvl)
#     #     out = data_consistency(k, k0, mask, self.noise_lvl)
#     #     x_res = ifft2(out) #[b,2,w,h]

#     #     if x.dim() == 4:
#     #         # x_res = x_res.permute(0, 3, 1, 2)
#     #         x_res = x_res
#     #     else:
#     #         raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

#     #     return x_res, out

#     def perform(self, x, k0, mask):
#         """
#         x, k0, mask - all in (n, 2, nx, ny) format
#         """
#         if x.dim() != 4:
#             raise ValueError("Input must be 4D!")
        
#         k = fft2(x)  # [b,2,h,w] -> [b,2,h,w]
#         out = data_consistency(k, k0, mask, self.noise_lvl)
#         x_res = ifft2(out)  # [b,2,h,w] -> [b,2,h,w]
        
#         return x_res, out


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - SR output in k-space [b,2,h,w]
    k0   - initially sampled LR elements in k-space [b,2,h,w]
    mask - sampling mask [b,2,h,w]
    noise_lvl - weight parameter n (controls trust in original data)
    
    For undersampled mask (mask has 0s and 1s):
        - Keep SR prediction where mask=0
        - Blend SR and original where mask=1
    
    For full mask (all 1s):
        - Blend everywhere: out = (k + n*k0) / (1 + n)
        - n controls trust: n=0 → full SR, n→∞ → full original
    """
    n = noise_lvl if noise_lvl is not None else 1.0
    
    # Weighted combination
    out = (1 - mask) * k + mask * (k + n * k0) / (1 + n)
    
    # # Simplified (works for both partial and full mask):
    # out = (k + mask * n * k0) / (1 + mask * n)
    
    return out


class DataConsistencyInKspace_I(nn.Module):
    """ Create data consistency operator
    
    Args:
        noise_lvl (float): Weight parameter 'n' for blending SR and original k-space
                          - n=0: Trust SR completely
                          - n=1: Equal weight (default)
                          - n>1: Trust original data more
    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_I, self).__init__()
        self.noise_lvl = noise_lvl if noise_lvl is not None else 1.0

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - SR output in image domain [b,2,h,w]
        k0   - initially sampled LR elements in k-space [b,2,h,w]
        mask - sampling mask [b,2,h,w]
        
        Returns:
            x_res - DC-corrected image [b,2,h,w]
            out   - DC-corrected k-space [b,2,h,w]
        """
        if x.dim() != 4:
            raise ValueError("Input must be 4D [b,2,h,w]!")
        
        # Transform to k-space
        k = fft2(x)  # [b,2,h,w] -> [b,2,h,w]
        
        # Data consistency: (k_sr + n*k_lr) / (1 + n) where mask=1
        out = data_consistency(k, k0, mask, self.noise_lvl)
        
        # Transform back to image domain
        x_res = ifft2(out)  # [b,2,h,w] -> [b,2,h,w]
        
        return x_res, out

class DataConsistencyInKspace_K(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace_K, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    # def perform(self, k, k0, mask):
    #     """
    #     k    - input in frequency domain, of shape (n, 2, nx, ny)
    #     k0   - initially sampled elements in k-space
    #     dc_mask - corresponding nonzero location
    #     """

    #     if k.dim() == 4:  # input is 2D [b,2,w,h]
    #         k = k.permute(0, 2, 3, 1) #[b,w,h,2]
    #     else:
    #         raise ValueError("error in data consistency layer!")

    #     out = data_consistency(k, k0, mask, self.noise_lvl) #[b,2,w,h]
    #     x_res = ifft2(out) #[b,2,w,h]
    #     # ========
    #     # ks_net_fin_out = x_res.cpu().detach().numpy()
    #     # sio.savemat('ks_net_fin_out.mat', {'data': ks_net_fin_out});
    #     # ========

    #     if k.dim() == 4:
    #         # x_res = x_res.permute(0, 3, 1, 2)
    #         x_res = x_res
    #     else:
    #         raise ValueError("Iuput dimension is wrong, it has to be a 2D input!")

    #     return x_res, out
    def perform(self, x, k0, mask):
        """
        x, k0, mask - all in (n, 2, nx, ny) format
        """
        if x.dim() != 4:
            raise ValueError("Input must be 4D!")
        
        k = fft2(x)  # [b,2,h,w] -> [b,2,h,w]
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = ifft2(out)  # [b,2,h,w] -> [b,2,h,w]
        
        return x_res, out

# Basic functions / transforms
def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


# def fft2(data):
#     """
#     Apply centered 2 dimensional Fast Fourier Transform.
#     input 2, w, h
#     output w, h, 2
#     Args:
#         data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
#             -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
#             assumed to be batch dimensions.[w,h,2]
#     Returns:
#         torch.Tensor: The FFT of the input.
#     """
#     # data = data.permute(1,2,0) #[w,h,2]

#     # assert data.size(-1) == 2
#     data = ifftshift(data, dim=(-3, -2))
#     data = torch.fft(data, 2, normalized=False)
#     data = fftshift(data, dim=(-3, -2))

#     # data = data.permute(2,0,1) #[2,w,h]
#     return data

def fft2_net(data):
    """
    Apply centered 2D FFT.
    
    Args:
        data: [B, 2, H, W] where data[:, 0] = real, data[:, 1] = imag
    
    Returns:
        kspace: [B, 2, H, W] k-space (real, imag)
    """
    # [B, 2, H, W] → [B, H, W] complex
    data_complex = torch.complex(data[:, 0], data[:, 1])
    
    # 2D FFT with centered k-space
    kspace_complex = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(data_complex, dim=(-2, -1)),
            norm='ortho'  # 推荐用 ortho，与 IFFT 对称
        ),
        dim=(-2, -1)
    )
    
    # [B, H, W] complex → [B, 2, H, W]
    kspace = torch.stack([kspace_complex.real, kspace_complex.imag], dim=1)
    
    return kspace


# def ifft2(data):
#     """
#     Apply centered 2-dimensional Inverse Fast Fourier Transform.
#     Args:
#         data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
#             -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
#             assumed to be batch dimensions.
#     Returns:
#         torch.Tensor: The IFFT of the input [b,2,w,h].
#     """
#     # assert data.size(-1) == 2
#     data = data.permute(0,2,3,1)#[0,w,h,2]
#     data = ifftshift(data, dim=(-3, -2))
#     data = torch.ifft(data, 2, normalized=False)
#     data = fftshift(data, dim=(-3, -2))
#     data = data.permute(0, 3, 1, 2)  # [0,2,w,h]
#     return data

def fft2(data):
    """
    Centered 2D FFT. Input: [b,2,h,w], Output: [b,2,h,w]
    """
    data = data.permute(0, 2, 3, 1).contiguous()   # [b,h,w,2]
    data_c = torch.view_as_complex(data)           # complex [b,h,w]

    # centered FFT
    data_c = torch.fft.ifftshift(data_c, dim=(-2, -1))
    k_c = torch.fft.fft2(data_c, norm="ortho")
    k_c = torch.fft.fftshift(k_c, dim=(-2, -1))

    k = torch.view_as_real(k_c).permute(0, 3, 1, 2).contiguous()  # [b,2,h,w]
    return k


def ifft2(data):
    """
    Centered 2D inverse FFT. Input: [b,2,h,w], Output: [b,2,h,w]
    """
    data = data.permute(0, 2, 3, 1).contiguous()   # [b,h,w,2]
    k_c = torch.view_as_complex(data)

    # centered IFFT
    k_c = torch.fft.ifftshift(k_c, dim=(-2, -1))
    x_c = torch.fft.ifft2(k_c, norm="ortho")
    x_c = torch.fft.fftshift(x_c, dim=(-2, -1))

    x = torch.view_as_real(x_c).permute(0, 3, 1, 2).contiguous()  # [b,2,h,w]
    return x



def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_eval(data):
    assert data.size(1) == 2
    return (data[:, 0:1, :, :] ** 2 + data[:, 1:2, :, :] ** 2).sqrt()


def to_spectral_img(data):
    """
    Compute the spectral images of a kspace data
    with keeping each column for creation of one spectral image
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2

    spectral_vol = torch.zeros([data.size(-2), data.size(-2), data.size(-2)])

    for i in range(data.size(-2)):
        kspc1 = torch.zeros(data.size())
        kspc1[:, i, :] = data[:, i, :]
        img1 = ifft2(kspc1)
        img1_abs = complex_abs(img1)

        spectral_vol[i, :, :] = img1_abs

    return spectral_vol