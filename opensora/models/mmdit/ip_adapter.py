import torch
import torch.nn as nn
from typing import Optional

class IPAdapter(nn.Module):
    def __init__(self, hidden_size: int = 3072, num_heads: int = 24):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Projection layers for reference image features
        self.image_proj = nn.Linear(768, hidden_size)  # CLIP -> MMDiT hidden size
        
    def forward(self, ref_features: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        # Project reference features to match MMDiT dimensions
        ref_features = self.image_proj(ref_features)
        return ref_features * scale

class IPAdapterProcessor:
    def __init__(self, ip_adapter: IPAdapter):
        self.ip_adapter = ip_adapter
        
    def __call__(self, attn: nn.Module, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor, ref_features: Optional[torch.Tensor] = None, ip_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        # Process image stream
        img_mod1, img_mod2 = attn.img_mod(vec)
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = torch.chunk(img_qkv, 3, dim=-1)
        else:
            img_q = attn.img_attn.q_proj(img_modulated)
            img_k = attn.img_attn.k_proj(img_modulated)
            img_v = attn.img_attn.v_proj(img_modulated)
            
        # Process text stream
        txt_mod1, txt_mod2 = attn.txt_mod(vec)
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        if attn.txt_attn.fused_qkv:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = torch.chunk(txt_qkv, 3, dim=-1)
        else:
            txt_q = attn.txt_attn.q_proj(txt_modulated)
            txt_k = attn.txt_attn.k_proj(txt_modulated)
            txt_v = attn.txt_attn.v_proj(txt_modulated)
            
        # Inject reference features if provided
        if ref_features is not None:
            ref_proj = self.ip_adapter(ref_features, ip_scale)
            # Add to image stream's K and V
            img_k = img_k + ref_proj
            img_v = img_v + ref_proj
            
        # Run attention - use forward_with_qkv method instead of directly calling forward
        img_attn = attn.img_attn.forward_with_qkv(img_q, img_k, img_v, pe)
        txt_attn = attn.txt_attn.forward_with_qkv(txt_q, txt_k, txt_v, pe)
        
        # Apply MLP
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt