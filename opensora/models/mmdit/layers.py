# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from torch import Tensor, nn

from .math import attention, liger_rope, rope


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class LigerEmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        cos_list = []
        sin_list = []
        for i in range(n_axes):
            cos, sin = liger_rope(ids[..., i], self.axes_dim[i], self.theta)
            cos_list.append(cos)
            sin_list.append(sin)
        cos_emb = torch.cat(cos_list, dim=-1).repeat(1, 1, 2).contiguous()
        sin_emb = torch.cat(sin_list, dim=-1).repeat(1, 1, 2).contiguous()

        return (cos_emb, sin_emb)


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class FusedRMSNorm(RMSNorm):
    def forward(self, x: Tensor):
        return LigerRMSNormFunction.apply(
            x,
            self.scale,
            1e-6,
            0.0,
            "llama",
            False,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = FusedRMSNorm(dim)
        self.key_norm = FusedRMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, fused_qkv: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        head_dim = dim // num_heads

        if fused_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor = None) -> Tensor:
        if self.fused_qkv:
            qkv = self.qkv(x)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        else:
            q = rearrange(self.q_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            k = rearrange(self.k_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            v = rearrange(self.v_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
        q, k = self.norm(q, k, v)
        if not self.fused_qkv:
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x
        
    # Add an overloaded method that can handle pre-computed q, k, v
    def forward_with_qkv(self, q: Tensor, k: Tensor, v: Tensor, pe: Tensor = None) -> Tensor:
        # Here we assume q, k, v are already appropriately shaped
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        # attn is the DoubleStreamBlock;
        # process img and txt separately while both is influenced by text vec

        # vec will interact with image latent and text context
        img_mod1, img_mod2 = attn.img_mod(vec)  # get shift, scale, gate for each mod
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)

        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm as in SD3 paper
        if not attn.img_attn.fused_qkv:
            img_q = rearrange(img_q, "B L H D -> B H L D")
            img_k = rearrange(img_k, "B L H D -> B H L D")
            img_v = rearrange(img_v, "B L H D -> B H L D")

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            txt_q = rearrange(attn.txt_attn.q_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        if not attn.txt_attn.fused_qkv:
            txt_q = rearrange(txt_q, "B L H D -> B H L D")
            txt_k = rearrange(txt_k, "B L H D -> B H L D")
            txt_v = rearrange(txt_v, "B L H D -> B H L D")

        # run actual attention, image and text attention are calculated together by concat different attn heads
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        fused_qkv: bool = True,
    ):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(hidden_size)
        self.txt_norm1 = nn.LayerNorm(hidden_size)
        self.img_mod = Modulation(hidden_size, double=True)
        self.txt_mod = Modulation(hidden_size, double=True)
        self.img_attn = SelfAttention(
            hidden_size, num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv
        )
        self.txt_attn = SelfAttention(
            hidden_size, num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv
        )
        self.img_norm2 = nn.LayerNorm(hidden_size)
        self.txt_norm2 = nn.LayerNorm(hidden_size)
        self.img_mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))
        self.processor = None

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        ref_features: Tensor = None,
        ip_scale: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        if self.processor is not None:
            return self.processor(self, img, txt, vec, pe, ref_features, ip_scale)
            
        # Process image stream
        img_mod1, img_mod2 = self.img_mod(vec)
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        if self.img_attn.fused_qkv:
            img_qkv = self.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = torch.chunk(img_qkv, 3, dim=-1)
        else:
            img_q = self.img_attn.q_proj(img_modulated)
            img_k = self.img_attn.k_proj(img_modulated)
            img_v = self.img_attn.v_proj(img_modulated)
            
        # Process text stream
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        if self.txt_attn.fused_qkv:
            txt_qkv = self.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = torch.chunk(txt_qkv, 3, dim=-1)
        else:
            txt_q = self.txt_attn.q_proj(txt_modulated)
            txt_k = self.txt_attn.k_proj(txt_modulated)
            txt_v = self.txt_attn.v_proj(txt_modulated)
            
        # Inject reference features if provided
        if ref_features is not None:
            ref_proj = self.processor.ip_adapter(ref_features, ip_scale)
            # Add to image stream's K and V
            img_k = img_k + ref_proj
            img_v = img_v + ref_proj
            
        # Run attention
        img_attn = self.img_attn(img_q, img_k, img_v, pe)
        txt_attn = self.txt_attn(txt_q, txt_k, txt_v, pe)
        
        # Apply MLP
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        if attn.fused_qkv:
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        else:
            q = rearrange(attn.q_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            k = rearrange(attn.k_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            v, mlp = torch.split(attn.v_mlp(x_mod), [attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=attn.num_heads)

        q, k = attn.norm(q, k, v)
        if not attn.fused_qkv:
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        fused_qkv: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SelfAttention(
            hidden_size, num_heads, qkv_bias=True, fused_qkv=fused_qkv
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))
        self.processor = None

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        ref_features: Tensor = None,
        ip_scale: float = 1.0,
    ) -> Tensor:
        if self.processor is not None:
            return self.processor(self, x, vec, pe, ref_features, ip_scale)
            
        # Process input
        x_norm = self.norm1(x)
        
        if self.attn.fused_qkv:
            qkv = self.attn.qkv(x_norm)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
        else:
            q = self.attn.q_proj(x_norm)
            k = self.attn.k_proj(x_norm)
            v = self.attn.v_proj(x_norm)
            
        # Inject reference features if provided
        if ref_features is not None:
            ref_proj = self.processor.ip_adapter(ref_features, ip_scale)
            # Add to K and V
            k = k + ref_proj
            v = v + ref_proj
            
        # Run attention
        attn = self.attn(q, k, v, pe)
        
        # Apply MLP
        x = x + self.attn.proj(attn)
        x = x + self.mlp(self.norm2(x))
        
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
