import collections
import math
from itertools import repeat
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    From https://github.com/pprp/timm/blob/master/timm/layers/patch_embed.py
    """
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0]
        assert W == self.img_size[1]
        x = self.proj(x)  # convolution to make patches
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # Flatten 16x16 to -> 256 # BCHW -> BNC
        x = self.norm(x)
        return x


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        import functorch
        vmap = functorch.vmap
    except ImportError:
        if hasattr(torch, 'vmap'):
            vmap = torch.vmap
        else:
            assert False, "functorch or a version of torch with vmap is required for FlexiViT resizing."

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    return v_resample_kernel(patch_embed)


class TimestepEmbedding(nn.Module):
    def __init__(self, frequency_embedding_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 768),
            nn.SiLU(),
            nn.Linear(768, 768)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t = self.timestep_embedding(t, self.frequency_embedding_size)
        t = self.mlp(t)
        return t

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class LabelEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DiT(nn.Module):
    def __init__(self, label_dim=768):
        super().__init__()
        self.x_emb = PatchEmbedding(32, 2, 3)
        # nn.init.constant_(self.x_emb.bias, 0.0) # Set conv bias to 0 ? this is not needed ?
        self.y_emb = nn.Sequential(
            # nn.LayerNorm(label_dim),  # LayerNorm to normalize input
            nn.Linear(label_dim, 768),  # Simple Linear layer
        )
        self.t_emb = TimestepEmbedding()

    def forward(self, x, y, t):
        """
        :param x: Img
        :param t: Timestep
        :param y: label
        :return:
        """
        x = self.x_emb(x)
        t = self.t_emb(t)
        y = self.y_emb(y) # Label
        # F.rms_norm()
        adaln_input = t + y # Timestep

        return x
