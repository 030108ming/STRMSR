import torch
import torch.nn as nn

from networks.network_STRMSR import STRMSR

def set_gpu(network, gpu_ids):
    """Move network to GPU and enable DataParallel."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this training code requires CUDA GPUs.")
    if not gpu_ids:
        raise RuntimeError("--gpu_ids must contain at least one visible CUDA device id.")

    visible_count = torch.cuda.device_count()
    invalid_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_count]
    if invalid_ids:
        raise RuntimeError(
            f"Invalid --gpu_ids {gpu_ids} with visible cuda device_count={visible_count}. "
            "GPU IDs are indexed inside CUDA_VISIBLE_DEVICES; for example, if "
            "CUDA_VISIBLE_DEVICES=0 exposes one GPU, use --gpu_ids 0."
        )

    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)
    return network


def get_network(opts):
    """
    Factory function to create network based on opts.net_G
    
    Supported networks:
    - McMRSR: Multi-contrast MRI SR (no temporal memory)
    - STMISR: Spatio-temporal multi-contrast MRI SR (with temporal memory)
    """
    
    # Common network parameters
    common_params = {
        'upscale': opts.upscale,
        'img_size': (opts.height, opts.width),
        'window_size': opts.window_size,
        'img_range': 1.,
        'depths': [2, 2, 2, 2],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.67
    }

    # Create network based on type
    if opts.net_G == 'STRMSR':
        print("[INFO] Creating STRMSR (no temporal memory)")
        network = STRMSR(**common_params)
    else:
        raise ValueError(
            f"Unknown network: '{opts.net_G}'\n"
            f"Available options: ['STRMSR']"
        )
    
    # Print parameter count
    num_param = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"[INFO] Network parameters: {num_param:,}")
    
    return set_gpu(network, opts.gpu_ids)
