import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import functools
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) dc_mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention dc_mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x

class FAM(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(FAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out

class DPRB(nn.Module):
    def __init__(self, nf):
        super(DPRB, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_a = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_b = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lr, ref):
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out

class DPRB_same_scale(nn.Module):
    def __init__(self, nf):
        super(DPRB_same_scale, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_a = nn.Conv2d(nf, nf, 3, 1, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up_b = nn.Conv2d(nf, nf, 3, 1, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lr, ref):
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out

class Conv2D(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Conv2D, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)
        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))

        return fea_L1

class MAB(nn.Module):
    def __init__(self, nf, out_chl, n_blks, upscale=4):
        super(MAB, self).__init__()
        block = functools.partial(ResidualBlock, nf=nf)

        ### spatial adaptation block ##
        self.FAM = FAM(nf, use_residual=True, learnable=True)
        ### joint residual feature aggregation block ##
        self.DPRB = DPRB(nf)
        self.DPRB_same_scale = DPRB_same_scale(nf)

        self.blk_x1 = make_layer(block, n_blks[3])
        self.blk_x2 = make_layer(block, n_blks[4])
        self.blk_x4 = make_layer(functools.partial(ResidualBlock, nf=nf), n_blks[5])

        self.conv_out = nn.Conv2d(nf, out_chl, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, tar_lr, F_M, upscale):

        if upscale == 1:
            # upscale ==1 only for ablation study
            # F_M[0]: [B, nf, H, W]，来自 ref_feature[0]
            warp_ref_x1 = self.FAM(tar_lr, F_M[0])
            fea_x1 = self.DPRB_same_scale(tar_lr, warp_ref_x1)
            fea_x1 = self.blk_x1(fea_x1)
            out = self.conv_out(fea_x1)
            out = torch.sigmoid(out)

        if upscale == 2:
            warp_ref_x1 = self.FAM(tar_lr, F_M[1])
            fea_x1 = self.DPRB_same_scale(tar_lr, warp_ref_x1)
            fea_x1 = self.blk_x1(fea_x1)
            fea_x1_up = F.interpolate(fea_x1, scale_factor=2, mode='bilinear', align_corners=False)

            warp_ref_x2 = self.FAM(fea_x1_up, F_M[0])
            fea_x2 = self.DPRB(fea_x1, warp_ref_x2)
            fea_x2 = self.blk_x2(fea_x2)

            out = self.conv_out(fea_x2)
            out = torch.sigmoid(out)

        elif upscale == 4:
            warp_ref_x1 = self.FAM(tar_lr, F_M[2])
            fea_x1 = self.DPRB_same_scale(tar_lr, warp_ref_x1)
            fea_x1 = self.blk_x1(fea_x1)
            fea_x1_up = F.interpolate(fea_x1, scale_factor=2, mode='bilinear', align_corners=False)

            warp_ref_x2 = self.FAM(fea_x1_up, F_M[1])
            fea_x2 = self.DPRB(fea_x1, warp_ref_x2)
            fea_x2 = self.blk_x2(fea_x2)
            fea_x2_up = F.interpolate(fea_x2, scale_factor=2, mode='bilinear', align_corners=False)

            warp_ref_x4 = self.FAM(fea_x2_up, F_M[0])
            fea_x4 = self.DPRB(fea_x2, warp_ref_x4)
            fea_x4 = self.blk_x4(fea_x4)

            out = self.conv_out(fea_x4)
            out = torch.sigmoid(out)

        return out


class PDFA(nn.Module):
    """Patch-wise Dynamic Feature Aggregation across reference views."""

    def __init__(self, C, patch_size=8, hidden=64):
        super().__init__()
        self.patch_size = patch_size
        
        # MLP/FCN to compute score from block descriptor
        self.score_net = nn.Sequential(
            nn.Linear(C, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, feats_list):
        if len(feats_list) == 1:
            return feats_list[0]

        B, C, H, W = feats_list[0].shape
        K = len(feats_list)
        ps = self.patch_size

        # Pad
        H0, W0 = H, W
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            feats_list = [F.pad(f, (0, pad_w, 0, pad_h), mode='replicate') for f in feats_list]
            H, W = H + pad_h, W + pad_w

        # Stack: [B, K, C, H, W]
        stack = torch.stack(feats_list, dim=1)
        flat = stack.view(B * K, C, H, W)

        # Get block descriptor: [B*K, C, py, px]
        desc = F.avg_pool2d(flat, kernel_size=ps, stride=ps)
        py, px = desc.shape[-2], desc.shape[-1]

        # ===== Use MLP to compute score =====
        desc_flat = desc.permute(0, 2, 3, 1).contiguous().view(B * K * py * px, C)
        logits_flat = self.score_net(desc_flat)
        logits = logits_flat.view(B, K, py, px, 1).permute(0, 1, 4, 2, 3)

        # Softmax across views
        w_blk = torch.softmax(logits, dim=1)  # [B, K, 1, py, px]

        # Expand to pixel-level weights
        w_map = w_blk.repeat_interleave(ps, dim=3).repeat_interleave(ps, dim=4)

        # Weighted sum
        out = (stack * w_map).sum(dim=1)

        # Crop back
        if pad_h or pad_w:
            out = out[:, :, :H0, :W0]
        
        return out


class STRMSR(nn.Module):
    r""" STRMSR
        A PyTorch impl of : `Transformer-empowered Multi-scale Contextual Matching and Aggregation for Multi-contrast MRI Super-resolution`, based on SwinIR.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 2
        embed_dim (int): Patch embedding dimension. Default: 60
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/4 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
    """
 # change the in_chans to 3 for MRI. from 2 to 3 
    def __init__(self, img_size=64, patch_size=1, in_chans=2,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1.,
                 **kwargs):
        super(STRMSR, self).__init__()
        num_in_ch = in_chans

        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = window_size
        n_blks = [2, 2, 2]
        n_blks_dec = [2, 2, 2, 12, 8, 4]
        self.lr_block_size = 8 # original 8
        self.ref_down_block_size = 1.5
        self.dilations = [1, 2, 3]
        self.num_nbr = 1
        self.psize = 3

        self.MAB = MAB(embed_dim, num_in_ch, n_blks=n_blks_dec, upscale=self.upscale)

        #####################################################################################################
        ################################### 1, Tar/Ref LR feature extraction ###################################
        self.conv2d_lr = Conv2D(in_chl=num_in_ch, nf=embed_dim, n_blks=n_blks)

        #####################################################################################################
        ################################### 2, Reference feature extraction ###################################
        self.conv_second_lr = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)
        self.conv_third_lr = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)

        self.conv_first_hr = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_second_hr = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)
        self.conv_third_hr = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)

        #####################################################################################################
        ################################### 3, deep feature extraction (STG) ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed_lr = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed_lr.num_patches
        patches_resolution = self.patch_embed_lr.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed_lr = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # split image into non-overlapping patches
        self.patch_embed_hr = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        # merge non-overlapping patches into image
        self.patch_unembed_hr = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers_lr = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         )
            self.layers_lr.append(layer)
        self.norm_lr = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction (Conv2d)
        self.conv_after_RSTB_lr = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        ################################### HR ###################################
        # build Residual Swin Transformer blocks (RSTB)
        self.layers_hr = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         )
            self.layers_hr.append(layer)
        self.norm_hr = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction (Conv2d)
        self.conv_after_RSTB_hr = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        ################################### 4, multi-scale feature aggregation ###################################
        # Add patch-wise aggregators

        if upscale == 1:
            self.view_agg_x1 = PDFA(embed_dim)
        elif upscale == 2:
            self.view_agg_x2 = PDFA(embed_dim)
            self.view_agg_x1 = PDFA(embed_dim)
        elif upscale == 4:
            self.view_agg_x4 = PDFA(embed_dim)
            self.view_agg_x2 = PDFA(embed_dim)
            self.view_agg_x1 = PDFA(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, C*k*k, H*W]
        # dim: scalar > 0
        # index: [N, Hi, Wi]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]  # views = [N, 1, -1]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1  # expanse = [-1, C*k*k, -1]
        index = index.clone().view(views).expand(expanse)  # [N, Hi, Wi] -> [N, 1, Hi*Wi] - > [N, C*k*k, Hi*Wi]
        return torch.gather(input, dim, index)  # [N, C*k*k, Hi*Wi]

    def search(self, lr, reflr, ks=3, pd=1, stride=1, dilations=[1, 2, 4]):
        # lr: [N, p*p, C, k_y, k_x]
        # reflr: [N, C, Hr, Wr]

        N, C, Hr, Wr = reflr.size()
        _, _, _, k_y, k_x = lr.size()
        x, y = k_x // 2, k_y // 2
        corr_sum = 0
        for i, dilation in enumerate(dilations):
            reflr_patches = F.unfold(reflr, kernel_size=(ks, ks), padding=dilation, stride=stride, dilation=dilation)  # [N, C*ks*ks, Hr*Wr]
            lr_patches = lr[:, :, :, y - dilation: y + dilation + 1: dilation,
                                     x - dilation: x + dilation + 1: dilation]  # [N, p*p, C, ks, ks]
            lr_patches = lr_patches.contiguous().view(N, -1, C * ks * ks)  # [N, p*p, C*ks*ks]

            lr_patches = F.normalize(lr_patches, dim=2)
            reflr_patches = F.normalize(reflr_patches, dim=1)
            corr = torch.bmm(lr_patches, reflr_patches)  # [N, p*p, Hr*Wr]
            corr_sum = corr_sum + corr

        sorted_corr, ind_l = torch.topk(corr_sum, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, p*p, num_nbr]

        return sorted_corr, ind_l

    def make_grid(self, idx_x1, idx_y1, diameter_x, diameter_y, s):
        idx_x1 = idx_x1 * s
        idx_y1 = idx_y1 * s
        idx_x1 = idx_x1.view(-1, 1).repeat(1, diameter_x * s)
        idx_y1 = idx_y1.view(-1, 1).repeat(1, diameter_y * s)
        idx_x1 = idx_x1 + torch.arange(0, diameter_x * s, dtype=torch.long, device=idx_x1.device).view(1, -1)
        idx_y1 = idx_y1 + torch.arange(0, diameter_y * s, dtype=torch.long, device=idx_y1.device).view(1, -1)

        ind_y_l = []
        ind_x_l = []
        for i in range(idx_x1.size(0)):
            grid_y, grid_x = torch.meshgrid(idx_y1[i], idx_x1[i])
            ind_y_l.append(grid_y.contiguous().view(-1))
            ind_x_l.append(grid_x.contiguous().view(-1))
        ind_y = torch.cat(ind_y_l)
        ind_x = torch.cat(ind_x_l)

        return ind_y, ind_x

    def search_org(self, lr, reflr, ks=3, pd=1, stride=1):
        # lr: [N, C, H, W]
        # reflr: [N, C, Hr, Wr]

        batch, c, H, W = lr.size()
        _, _, Hr, Wr = reflr.size()

        reflr_unfold = F.unfold(reflr, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, Hr*Wr]
        lr_unfold = F.unfold(lr, kernel_size=(ks, ks), padding=0, stride=stride)
        lr_unfold = lr_unfold.permute(0, 2, 1)  # [N, H*W, C*k*k]

        lr_unfold = F.normalize(lr_unfold, dim=2)
        reflr_unfold = F.normalize(reflr_unfold, dim=1)

        corr = torch.bmm(lr_unfold, reflr_unfold)  # [N, H*W, Hr*Wr]
        corr = corr.view(batch, H-2, W-2, (Hr-2)*(Wr-2))
        sorted_corr, ind_l = torch.topk(corr, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, H, W, num_nbr]

        return sorted_corr, ind_l


    def transfer(self, fea, index, soft_att, ks=3, pd=1, stride=1):
        # fea: [N, C, H, W]
        # index: [N, Hi, Wi]
        # soft_att: [N, 1, Hi, Wi]
        scale = stride

        fea_unfold = F.unfold(fea, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, H*W]
        out_unfold = self.bis(fea_unfold, 2, index)  # [N, C*k*k, Hi*Wi]
        divisor = torch.ones_like(out_unfold)

        _, Hi, Wi = index.size()
        out_fold = F.fold(out_unfold, output_size=(Hi*scale, Wi*scale), kernel_size=(ks, ks), padding=pd, stride=stride)
        divisor = F.fold(divisor, output_size=(Hi*scale, Wi*scale), kernel_size=(ks, ks), padding=pd, stride=stride)
        soft_att_resize = F.interpolate(soft_att, size=(Hi*scale, Wi*scale), mode='bilinear',align_corners=True)
        out_fold = out_fold / divisor * soft_att_resize
        # out_fold = out_fold / (ks*ks) * soft_att_resize
        return out_fold

    def forward_features_RSTB(self, x, branch="lr"):
        """
        branch:
            "lr" -> use LR-RSTB (shared by tar + refLR)
            "hr" -> use HR-RSTB (only for refHR)
        """
        # original spatial size
        _, _, h, w = x.size()

        # compute pad to make multiple of window_size
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')

        # new size after padding
        x_size = (x.shape[2], x.shape[3])

        # ===== select branch modules =====
        if branch == "lr":
            patch_embed = self.patch_embed_lr
            patch_unembed = self.patch_unembed_lr
            layers = self.layers_lr
            norm = self.norm_lr
        elif branch == "hr":
            patch_embed = self.patch_embed_hr
            patch_unembed = self.patch_unembed_hr
            layers = self.layers_hr
            norm = self.norm_hr
        else:
            raise ValueError(f"Unknown branch={branch}, must be 'lr' or 'hr'")

        # ===== standard SwinIR flow =====
        x = patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in layers:
            x = layer(x, x_size)

        x = norm(x)
        x = patch_unembed(x, x_size)

        # crop back to original size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = x[:, :, :h, :w]

        return x


    def local_search_and_warp(
        self,
        tar_lr_pad, ref_lr_pad, ref_feat_pad,
        center_x_init, center_y_init,
        py, px, k_y, k_x, radius, N,
    ):
        """
        Local NCC search around init center for each block, then transfer warp.
        
        Key design:
            crop_size = 2*radius + 3   (includes +2 border for ks=3)
            valid plane = crop_size - 2 = 2*radius + 1
        """
        device = tar_lr_pad.device
        _, C, H, W = tar_lr_pad.shape
        M = N * py * px

        # Pure-radius crop definition
        crop_size = 2 * radius + 3
        w_valid = crop_size - 2  # = 2*radius + 1
        h_valid = crop_size - 2

        # Top-left of crop from init centers
        crop_x1 = torch.round(center_x_init - (crop_size // 2)).long()
        crop_y1 = torch.round(center_y_init - (crop_size // 2)).long()
        crop_x1 = crop_x1.clamp(0, W - crop_size)
        crop_y1 = crop_y1.clamp(0, H - crop_size)

        # Gather ref crops
        ind_y, ind_x = self.make_grid(crop_x1, crop_y1, crop_size, crop_size, s=1)
        ind_b = torch.repeat_interleave(
            torch.arange(0, N, dtype=torch.long, device=device),
            py * px * crop_size * crop_size
        )

        ref_lr_crops = ref_lr_pad[ind_b, :, ind_y, ind_x].view(
            M, crop_size, crop_size, C
        ).permute(0, 3, 1, 2).contiguous()

        ref_feat_crops = ref_feat_pad[ind_b, :, ind_y, ind_x].view(
            M, crop_size, crop_size, C
        ).permute(0, 3, 1, 2).contiguous()

        # Extract target patches aligned with blocks
        tar_patches = F.pad(tar_lr_pad, pad=(1, 1, 1, 1), mode='replicate')
        tar_patches = F.unfold(tar_patches, kernel_size=(k_y + 2, k_x + 2), stride=(k_y, k_x))
        tar_patches = tar_patches.view(N, C, k_y + 2, k_x + 2, py * px).permute(0, 4, 1, 2, 3)
        tar_patches = tar_patches.contiguous().view(M, C, k_y + 2, k_x + 2)

        # Dense local NCC
        corr, idx = self.search_org(
            tar_patches, ref_lr_crops, ks=self.psize, pd=self.psize // 2, stride=1
        )
        index_all = idx[:, :, :, 0]
        soft_att = corr[:, :, :, 0:1].permute(0, 3, 1, 2)

        # Warp ref features
        warp_patches = self.transfer(
            ref_feat_crops, index_all, soft_att,
            ks=self.psize, pd=self.psize // 2, stride=1
        )
        warp_patches = warp_patches.view(N, py, px, C, k_y, k_x).permute(0, 3, 1, 4, 2, 5).contiguous()
        warp_out = warp_patches.view(N, C, py * k_y, px * k_x)

        # Refined absolute coords for center propagation
        x_in = index_all % w_valid
        y_in = torch.div(index_all, w_valid, rounding_mode='floor')

        crop_x1_view = crop_x1.view(M, 1, 1)
        crop_y1_view = crop_y1.view(M, 1, 1)

        abs_x = crop_x1_view + 1 + x_in
        abs_y = crop_y1_view + 1 + y_in

        # Center pick: middle pixel of block
        center_x = abs_x[:, k_y // 2, k_x // 2].float()
        center_y = abs_y[:, k_y // 2, k_x // 2].float()

        center_x_grid = center_x.view(N, py, px)
        center_y_grid = center_y.view(N, py, px)

        return warp_out, center_x_grid, center_y_grid


    def contextual_matching_coarse_to_fine(self, tar_lr_pyramid, ref_lr_pyramid, ref_feature, upscale):
        """
        Coarse-to-fine contextual matching with STRICT pyramid sizing.
        
        Pyramid structure:
            x1 (H/4): pad to multiple of 8 => (H2p, W2p)
            x2 (H/2): force to (2*H2p, 2*W2p)
            x4 (H):   force to (4*H2p, 4*W2p)

        Args:
            tar_lr_pyramid: [tar_lr0, tar_lr1, tar_lr2]  # [H, H/2, H/4]
            ref_lr_pyramid: [ref_lr0, ref_lr1, ref_lr2]  # [H, H/2, H/4]
            ref_feature:    [ref_x4, ref_x2, ref_x1]     # [H, H/2, H/4]
            upscale: int (1/2/4)

        Returns:
            F_M: warped features at each scale
        """
        assert upscale in [1, 2, 4]

        tar_lr0, tar_lr1, tar_lr2 = tar_lr_pyramid  # H, H/2, H/4
        ref_lr0, ref_lr1, ref_lr2 = ref_lr_pyramid
        ref_x4, ref_x2, ref_x1 = ref_feature        # H, H/2, H/4

        device = tar_lr2.device
        N, C, h2, w2 = tar_lr2.shape
        h1, w1 = tar_lr1.shape[2], tar_lr1.shape[3]
        h0, w0 = tar_lr0.shape[2], tar_lr0.shape[3]

        k_y = self.lr_block_size  # 8
        k_x = self.lr_block_size  # 8

        # ================================================================
        # Helper functions
        # ================================================================
        def _ceil_to_mul(val, block):
            return val + (block - val % block) % block

        def _pad_to_hw(x, Ht, Wt):
            """Pad tensor to exact (Ht, Wt) size."""
            _, _, H, W = x.shape
            pad_h = max(0, Ht - H)
            pad_w = max(0, Wt - W)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
            return x[:, :, :Ht, :Wt]

        # ================================================================
        # (1) Compute STRICT padded sizes based on x1 level (H/4)
        # ================================================================
        H2p = _ceil_to_mul(h2, k_y)  # e.g., 48
        W2p = _ceil_to_mul(w2, k_x)  # e.g., 64
        H1p, W1p = 2 * H2p, 2 * W2p  # e.g., 96, 128
        H0p, W0p = 4 * H2p, 4 * W2p  # e.g., 192, 256

        # ================================================================
        # (2) Force each level into strict sizes
        # ================================================================
        tar2p = _pad_to_hw(tar_lr2, H2p, W2p)
        ref2p = _pad_to_hw(ref_lr2, H2p, W2p)
        refx1p = _pad_to_hw(ref_x1, H2p, W2p)

        tar1p = _pad_to_hw(tar_lr1, H1p, W1p)
        ref1p = _pad_to_hw(ref_lr1, H1p, W1p)
        refx2p = _pad_to_hw(ref_x2, H1p, W1p)

        tar0p = _pad_to_hw(tar_lr0, H0p, W0p)
        ref0p = _pad_to_hw(ref_lr0, H0p, W0p)
        refx4p = _pad_to_hw(ref_x4, H0p, W0p)

        # Block grids
        py2, px2 = H2p // k_y, W2p // k_x  # e.g., 6, 8
        py1, px1 = H1p // k_y, W1p // k_x  # e.g., 12, 16
        py0, px0 = H0p // k_y, W0p // k_x  # e.g., 24, 32

        # ================================================================
        # (3) Level-2 (x1): Coarse global search + in-crop dense NCC
        # ================================================================
        diameter_x = 2 * int(W2p // (2 * px2) * self.ref_down_block_size) + 1
        diameter_y = 2 * int(H2p // (2 * py2) * self.ref_down_block_size) + 1

        # Target block patches (10x10 with border)
        lr_patches2 = F.pad(tar2p, pad=(1, 1, 1, 1), mode='replicate')
        lr_patches2 = F.unfold(lr_patches2, kernel_size=(k_y + 2, k_x + 2), stride=(k_y, k_x))
        lr_patches2 = lr_patches2.view(N, C, k_y + 2, k_x + 2, py2 * px2).permute(0, 4, 1, 2, 3)

        # Coarse search with dilations
        sorted_corr2, ind_l2 = self.search(
            lr_patches2, ref2p, ks=3, pd=1, stride=1, dilations=self.dilations
        )

        # Compute crop top-left in ref2p
        index2 = ind_l2[:, :, 0]  # [N, py2*px2]
        idx_x = index2 % W2p
        idx_y = torch.div(index2, W2p, rounding_mode='floor')

        idx_x1 = (idx_x - diameter_x // 2 - 1).clamp(0, W2p - (diameter_x + 2))
        idx_y1 = (idx_y - diameter_y // 2 - 1).clamp(0, H2p - (diameter_y + 2))

        # Crop ref patches
        ind_y2, ind_x2 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 1)
        M2 = N * py2 * px2
        ind_b2 = torch.repeat_interleave(
            torch.arange(0, N, dtype=torch.long, device=device),
            py2 * px2 * (diameter_y + 2) * (diameter_x + 2)
        )

        reflr_crop2 = ref2p[ind_b2, :, ind_y2, ind_x2].view(
            M2, diameter_y + 2, diameter_x + 2, C
        ).permute(0, 3, 1, 2).contiguous()

        refx1_crop2 = refx1p[ind_b2, :, ind_y2, ind_x2].view(
            M2, diameter_y + 2, diameter_x + 2, C
        ).permute(0, 3, 1, 2).contiguous()

        # Dense NCC inside crop
        tar_patch2 = lr_patches2.contiguous().view(M2, C, k_y + 2, k_x + 2)
        corr2, idx2 = self.search_org(tar_patch2, reflr_crop2, ks=self.psize, pd=self.psize // 2, stride=1)

        index_all_2 = idx2[:, :, :, 0]  # [M2, 8, 8]
        soft_att_2 = corr2[:, :, :, 0:1].permute(0, 3, 1, 2)  # [M2, 1, 8, 8]

        # Warp x1 features
        warp_patch_x1 = self.transfer(
            refx1_crop2, index_all_2, soft_att_2,
            ks=self.psize, pd=self.psize // 2, stride=1
        )
        warp_patch_x1 = warp_patch_x1.view(N, py2, px2, C, k_y, k_x).permute(0, 3, 1, 4, 2, 5).contiguous()
        warp_x1 = warp_patch_x1.view(N, C, H2p, W2p)

        # Compute absolute coords for center extraction
        w_valid2 = diameter_x
        x_in2 = index_all_2 % w_valid2
        y_in2 = torch.div(index_all_2, w_valid2, rounding_mode='floor')

        idx_x1_view = idx_x1.reshape(-1, 1, 1)
        idx_y1_view = idx_y1.reshape(-1, 1, 1)

        abs_x2 = idx_x1_view + 1 + x_in2
        abs_y2 = idx_y1_view + 1 + y_in2

        # Center: middle pixel of 8x8 block
        center_x2 = abs_x2[:, k_y // 2, k_x // 2].float()
        center_y2 = abs_y2[:, k_y // 2, k_x // 2].float()

        center_x2_grid = center_x2.view(N, py2, px2)
        center_y2_grid = center_y2.view(N, py2, px2)

        # ================================================================
        # Return for upscale == 1
        # ================================================================
        if upscale == 1:
            warp_x1_out = warp_x1[:, :, :h2, :w2]
            F_M = [warp_x1_out]
            return F_M

        # ================================================================
        # (4) Level-1 refine (x2): init from Level-2, scale x2
        #     Use nearest to preserve block boundaries
        # ================================================================
        init_x1_grid = F.interpolate(
            center_x2_grid.unsqueeze(1), size=(py1, px1), mode='nearest'
        ).squeeze(1) * 2.0
        init_y1_grid = F.interpolate(
            center_y2_grid.unsqueeze(1), size=(py1, px1), mode='nearest'
        ).squeeze(1) * 2.0

        M1 = N * py1 * px1
        init_x1 = init_x1_grid.reshape(M1)
        init_y1 = init_y1_grid.reshape(M1)

        warp_x2, center_x1_grid, center_y1_grid = self.local_search_and_warp(
            tar_lr_pad=tar1p, ref_lr_pad=ref1p, ref_feat_pad=refx2p,
            center_x_init=init_x1, center_y_init=init_y1,
            py=py1, px=px1, k_y=k_y, k_x=k_x, radius=4, N=N,
        )

        # ================================================================
        # Return for upscale == 2
        # ================================================================
        if upscale == 2:
            warp_x1_out = warp_x1[:, :, :h2, :w2]
            warp_x2_out = warp_x2[:, :, :h1, :w1]
            F_M = [warp_x2_out, warp_x1_out]
            return F_M

        # ================================================================
        # (5) Level-0 refine (x4): init from Level-1, scale x2
        # ================================================================
        init_x0_grid = F.interpolate(
            center_x1_grid.unsqueeze(1), size=(py0, px0), mode='nearest'
        ).squeeze(1) * 2.0
        init_y0_grid = F.interpolate(
            center_y1_grid.unsqueeze(1), size=(py0, px0), mode='nearest'
        ).squeeze(1) * 2.0

        M0 = N * py0 * px0
        init_x0 = init_x0_grid.reshape(M0)
        init_y0 = init_y0_grid.reshape(M0)

        warp_x4, _, _ = self.local_search_and_warp(
            tar_lr_pad=tar0p, ref_lr_pad=ref0p, ref_feat_pad=refx4p,
            center_x_init=init_x0, center_y_init=init_y0,
            py=py0, px=px0, k_y=k_y, k_x=k_x, radius=5, N=N,
        )

        # ================================================================
        # (6) Crop back to original sizes and return
        # ================================================================
        warp_x1_out = warp_x1[:, :, :h2, :w2]
        warp_x2_out = warp_x2[:, :, :h1, :w1]
        warp_x4_out = warp_x4[:, :, :h0, :w0]

        F_M = [warp_x4_out, warp_x2_out, warp_x1_out]

        return F_M


    def forward(self, tar, reflr, ref):
        """
        Forward pass with multi-view reference support.
        
        Args:
            tar: Target LR (B, 2, H, W)
            reflr: Reference LR - MULTI-VIEW! (B, num_views, 2, H, W)
            ref: Reference HR - MULTI-VIEW! (B, num_views, 2, H, W)
        
        Returns:
            Tar_Rec_SR: Reconstructed output (B, 2, H, W)
        """
        
        # ========== CHECK IF MULTI-VIEW OR SINGLE-VIEW ==========
        if reflr.dim() == 5:  # Multi-view: (B, num_views, 2, H, W)
            B, num_views, C, H, W = reflr.shape
            is_multiview = True
        else:  # Single-view: (B, 2, H, W)
            B, C, H, W = reflr.shape
            num_views = 1
            is_multiview = False
            # Add view dimension
            reflr = reflr.unsqueeze(1)
            ref = ref.unsqueeze(1)

        # ========== PROCESS TARGET ==========
        tar_lr0 = self.conv2d_lr(tar)          # [B, C, H, W]
        # tar_lr0 = self.conv_after_RSTB(self.forward_features_RSTB(tar_lr0)) + tar_lr0
        tar_lr1 = self.conv_second_lr(tar_lr0) # [B, C, H/2, W/2]
        tar_lr2 = self.conv_third_lr(tar_lr1)  # [B, C, H/4, W/4]
        tar_lr2 = self.conv_after_RSTB_lr(self.forward_features_RSTB(tar_lr2, branch="lr")) + tar_lr2

        tar_lr_pyramid = [tar_lr0, tar_lr1, tar_lr2]
        tar_lr = tar_lr2  # MAB input stays x1 level

        # ========== PROCESS EACH REFERENCE VIEW ==========
        all_F_M = []

        for view_idx in range(num_views):
            reflr_view = reflr[:, view_idx, :, :, :]
            ref_view   = ref[:, view_idx, :, :, :]

            # Ref LR pyramid (H, H/2, H/4)
            ref_lr0 = self.conv2d_lr(reflr_view)
            # ref_lr0 = self.conv_after_RSTB(self.forward_features_RSTB(ref_lr0)) + ref_lr0
            ref_lr1 = self.conv_second_lr(ref_lr0)
            ref_lr2 = self.conv_third_lr(ref_lr1)
            ref_lr2 = self.conv_after_RSTB_lr(self.forward_features_RSTB(ref_lr2, branch="lr")) + ref_lr2
            ref_lr_pyramid = [ref_lr0, ref_lr1, ref_lr2]

            # Ref HR feature pyramid for warping: [H, H/2, H/4]
            ref_0 = self.conv_first_hr(ref_view)  # H
            ref_0 = self.conv_after_RSTB_hr(self.forward_features_RSTB(ref_0, branch="hr")) + ref_0
            ref_1 = self.conv_second_hr(ref_0)    # H/2
            ref_1 = self.conv_after_RSTB_hr(self.forward_features_RSTB(ref_1, branch="hr")) + ref_1
            ref_2 = self.conv_third_hr(ref_1)     # H/4
            ref_2 = self.conv_after_RSTB_hr(self.forward_features_RSTB(ref_2, branch="hr")) + ref_2
            ref_feature = [ref_0, ref_1, ref_2]  # == [ref_x4, ref_x2, ref_x1] in your coarse_to_fine

            #  Coarse-to-fine refine matching
            F_M = self.contextual_matching_coarse_to_fine(
                tar_lr_pyramid=tar_lr_pyramid,
                ref_lr_pyramid=ref_lr_pyramid,
                ref_feature=ref_feature,
                upscale=self.upscale
            )
            all_F_M.append(F_M)

        # ===== Patch-wise dynamic feature aggregation across views =====
        if num_views > 1:
            if self.upscale == 1:
                F_M_agg = [self.view_agg_x1([f[0] for f in all_F_M])]
            elif self.upscale == 2:
                F_M_agg = [
                    self.view_agg_x2([f[0] for f in all_F_M]),
                    self.view_agg_x1([f[1] for f in all_F_M]),
                ]
            elif self.upscale == 4:
                F_M_agg = [
                    self.view_agg_x4([f[0] for f in all_F_M]),
                    self.view_agg_x2([f[1] for f in all_F_M]),
                    self.view_agg_x1([f[2] for f in all_F_M]),
                ]
        else:
            F_M_agg = all_F_M[0]

        # Multi-scale aggregation
        Tar_Rec_SR = self.MAB(tar_lr, F_M_agg, self.upscale)
        
        return Tar_Rec_SR


    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed_lr.flops()  
        for i, layer in enumerate(self.layers_lr):  
            flops += layer.flops()
        #  HR 
        for i, layer in enumerate(self.layers_hr):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops

if __name__ == '__main__':
    upscale = 1
    window_size = 8
    height = 176
    width = 256
    model = STRMSR(upscale=upscale, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    num_param = sum([p.numel() for p in model.parameters() if p.requires_grad])

    print(model)
    print('Number of parameters: {}'.format(num_param))
    x = torch.randn((1, 2,2, height, width))
    x_T1_lr = torch.randn((1, 3, 2, height, width))
    x_T1 = torch.randn((1, 3, 2, height*upscale, width*upscale))
    x = model(x,x_T1_lr,x_T1)
    print(x.shape)

