import math
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Dict, Tuple, Optional # Added Optional type hinting

import torch
from einops import rearrange, repeat
from mmengine.config import Config
from peft import PeftModel
from torch import Tensor, nn
import numpy as np # Added for potential embedding generation

from opensora.datasets.aspect import get_image_size
from opensora.models.mmdit.model import MMDiTModel
from opensora.models.text.conditioner import HFEmbedder
from opensora.registry import MODELS, build_module
from opensora.utils.inference import (
    SamplingMethod,
    collect_references_batch, # Kept for standard I2V
    prepare_inference_condition, # Kept for standard I2V
)
from opensora.utils.logger import log_message # Added for logging


# ======================================================
# Sampling Options (Unchanged)
# ======================================================
import torchvision.transforms as T
from PIL import Image

def load_image(path, device, dtype, height, width, num_frames=16):
    image = Image.open(path).convert("RGB")
    preprocess = T.Compose([
        T.Resize((height, width)),
        T.ToTensor()
    ])
    image = preprocess(image).unsqueeze(0).unsqueeze(2)  
    image = image.repeat(1, 1, num_frames, 1, 1)
    image = image.to(device, dtype)
    return image


def personalize_latent(z, noise_level=0.0):
    if noise_level > 0:
        noise = torch.randn_like(z)
        z = (1 - noise_level) * z + noise_level * noise
    return z


@dataclass
class SamplingOption:
    # The width of the image/video.
    width: int | None = None
    # The height of the image/video.
    height: int | None = None
    # The resolution of the image/video. If provided, it will override the height and width.
    resolution: str | None = None
    # The aspect ratio of the image/video. If provided, it will override the height and width.
    aspect_ratio: str | None = None
    # The number of frames.
    num_frames: int = 1
    # The number of sampling steps.
    num_steps: int = 50
    # The classifier-free guidance (text).
    guidance: float = 4.0
    # use oscillation for text guidance
    text_osci: bool = False
    # The classifier-free guidance (image), or for the guidance on condition for i2v and v2v
    guidance_img: float | None = None
    # use oscillation for image guidance
    image_osci: bool = False
    # use temporal scaling for image guidance
    scale_temporal_osci: bool = False
    # The seed for the random number generator.
    seed: int | None = None
    # Whether to shift the schedule.
    shift: bool = True
    # The sampling method.
    method: str | SamplingMethod = SamplingMethod.I2V
    # Temporal reduction
    temporal_reduction: int = 1
    # is causal vae
    is_causal_vae: bool = False
    # flow shift
    flow_shift: float | None = None


def sanitize_sampling_option(sampling_option: SamplingOption) -> SamplingOption:
    """
    Sanitize the sampling options.
    """
    if (
        sampling_option.resolution is not None
        or sampling_option.aspect_ratio is not None
    ):
        assert (
            sampling_option.resolution is not None
            and sampling_option.aspect_ratio is not None
        ), "Both resolution and aspect ratio must be provided"
        resolution = sampling_option.resolution
        aspect_ratio = sampling_option.aspect_ratio
        height, width = get_image_size(resolution, aspect_ratio, training=False)
    else:
        assert (
            sampling_option.height is not None and sampling_option.width is not None
        ), "Both height and width must be provided"
        height, width = sampling_option.height, sampling_option.width

    # Ensure divisible by VAE downscale factor and patch size (typically 16 * patch_size)
    # Assuming patch_size = 2 and VAE spatial downscale = 8 => 16
    # Let's just ensure divisibility by 16 for simplicity, adjust if needed
    downscale_factor = int(os.environ.get("AE_SPATIAL_COMPRESSION", 8)) # VAE spatial downscale
    patch_embed_factor = 16 # Assuming this is fixed based on VAE + patchify
    height = (height // patch_embed_factor + (1 if height % patch_embed_factor else 0)) * patch_embed_factor
    width = (width // patch_embed_factor + (1 if width % patch_embed_factor else 0)) * patch_embed_factor

    replace_dict = dict(height=height, width=width)

    if isinstance(sampling_option.method, str):
        try:
            # Check if it's a standard SamplingMethod enum
            method = SamplingMethod(sampling_option.method)
            replace_dict["method"] = method
        except ValueError:
            # Keep as string if it's a custom key like "personalize"
            pass

    return replace(sampling_option, **replace_dict)


def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    """
    get oscillation guidance for cfg.
    """
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs

# ======================================================
# Helper Functions for Personalization (NEW)
# ======================================================
def get_fixed_pseudo_concept_embedding(index: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Generates a fixed, unique embedding based on an index.
    Using sinusoidal encoding for better distinction.
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    pos = torch.tensor([index + 1], device=device, dtype=dtype) # Start index from 1
    pos_enc_a = torch.sin(pos[:, None] * inv_freq)
    pos_enc_b = torch.cos(pos[:, None] * inv_freq)
    # Handle odd dimensions
    if dim % 2 != 0:
        pos_enc_b = torch.cat([pos_enc_b, torch.zeros(1, 1, device=device, dtype=dtype)], dim=-1)
    embedding = torch.cat([pos_enc_a, pos_enc_b], dim=-1).unsqueeze(1) # Shape: [1, 1, Dim]

    return embedding * 0.1 # Scale down magnitude - IMPORTANT tuning parameter


# ======================================================
# Denoising Classes
# ======================================================

class Denoiser(ABC):
    @abstractmethod
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        """Denoise the input."""

    @abstractmethod
    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Tensor]]: # Modified return type
        """Prepare the guidance contexts for the model."""


# --- Keep Original I2VDenoiser for standard I2V conditioning ---
class I2VDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img") # packed noisy latent z
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance")
        guidance_img = kwargs.pop("guidance_img")

        # Standard I2V conditional inputs (masks/masked_ref for latent modification)
        masks = kwargs.pop("masks")         # packed masks
        masked_ref = kwargs.pop("masked_ref") # packed masked reference latent
        sigma_min = kwargs.pop("sigma_min") # Required for noise addition potentially

        # Other necessary inputs prepared by `prepare`
        txt_cond = kwargs.pop("txt_cond")
        txt_uncond = kwargs.pop("txt_uncond")
        txt_uncond_img = kwargs.pop("txt_uncond_img")
        pass_through_kwargs = {k: v for k, v in kwargs.items() if k not in ['guidance', 'guidance_img', 'timesteps', 'img']}

        # Oscillation args
        text_osci = kwargs.pop("text_osci", False)
        image_osci = kwargs.pop("image_osci", False)
        scale_temporal_osci = kwargs.pop("scale_temporal_osci", False)
        patch_size = kwargs.pop("patch_size", 2) # Get patch size used for packing

        z = img # Start with initial packed noisy latent

        # Denoising Loop for Standard I2V CFG
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # Prepare model input: [cond_z, uncond_z_img, uncond_z_text]
            # Where cond_z = z + masked_ref (potentially noised)
            # uncond_z_img = z (no text guidance)
            # uncond_z_text = z (no image guidance)

            # Add noise to reference latents based on current timestep (matching diffusion process)
            # noise = torch.randn_like(masked_ref)
            # noised_masked_ref = model.scheduler.add_noise(masked_ref, noise, t_curr) # Need access to scheduler
            # For simplicity, let's assume no additional noise injection here, relies on initial noise
            # This part might need refinement based on how exactly masked_ref is used
            noised_masked_ref = masked_ref # Placeholder

            # Create conditional latent (apply mask)
            cond_z = (1.0 - masks) * z + masks * noised_masked_ref

            model_input = torch.cat([cond_z, z, z], dim=0) # Input: [Conditional, Img_Uncond, Text_Uncond]
            t_vec = torch.full((model_input.shape[0],), t_curr, dtype=z.dtype, device=z.device)

            # Prepare combined context [cond_txt, uncond_img_txt, uncond_text_txt]
            combined_context = torch.cat([txt_cond, txt_uncond_img, txt_uncond], dim=0)
            model_kwargs = {**pass_through_kwargs, "txt": combined_context}
            # Duplicate other per-batch args
            for key in ['img_ids', 'txt_ids', 'y_vec']:
                 if key in model_kwargs:
                      key_val = model_kwargs[key]
                      model_kwargs[key] = torch.cat([key_val, key_val, key_val], dim=0)


            # Forward pass
            pred = model(
                img=model_input,
                timesteps=t_vec,
                **model_kwargs,
            )

            # CFG Calculation
            cond_pred, uncond_img_pred, uncond_text_pred = pred.chunk(3, dim=0)

            # Apply oscillation
            text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
            image_gs = get_oscillation_gs(guidance_img, i) if image_osci else guidance_img

            # Scale temporal oscillation needs original T, H, W -> Unpack/Pack is inefficient here
            # Skipping scale_temporal_osci logic for brevity, requires careful tensor shape management
            if image_gs > 1.0 and scale_temporal_osci:
                 log_message("Warning: scale_temporal_osci not fully implemented in this refactor.")
                 # Placeholder logic: needs unpacking, applying scale, repacking.
                 # image_gs = ... (calculate scaled image_gs based on unpacked shape)


            # Standard CFG formula for I2V
            guided_pred = uncond_text_pred + \
                          image_gs * (uncond_img_pred - uncond_text_pred) + \
                          text_gs * (cond_pred - uncond_img_pred)

            # Euler step
            z = z + (t_prev - t_curr) * guided_pred

        return z

    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Tensor]]:
        ret = {}
        model_t5 = optional_models.get("t5")
        batch_size = len(text)

        neg = kwargs.get("neg", None)
        guidance_img = kwargs.pop("guidance_img")
        ret["guidance_img"] = guidance_img # Pass guidance scale through

        if neg is None:
            neg = [""] * batch_size

        # Prepare 3 sets of text embeddings needed for I2V CFG:
        # 1. Conditional (Positive Text)
        # 2. Unconditional for Image Guidance (Negative Text)
        # 3. Unconditional for Text Guidance (Positive Text - used to subtract img-only guidance later)
        # Note: Open-Sora's original I2V might use different uncond scheme, this matches common practice.
        ret["txt_cond"] = model_t5(text).to(device, dtype)
        ret["txt_uncond_img"] = model_t5(neg).to(device, dtype) # Negative prompts for image-only guidance path
        ret["txt_uncond"] = model_t5(neg).to(device, dtype) # Negative prompts for final unconditional base

        # Ensure consistent sequence lengths if T5 varies output length
        max_len = max(ret["txt_cond"].shape[1], ret["txt_uncond_img"].shape[1], ret["txt_uncond"].shape[1])
        for key in ["txt_cond", "txt_uncond_img", "txt_uncond"]:
             if ret[key].shape[1] < max_len:
                 pad_len = max_len - ret[key].shape[1]
                 ret[key] = torch.cat([ret[key], torch.zeros(batch_size, pad_len, ret[key].shape[-1], device=device, dtype=dtype)], dim=1)

        return text, ret # Return original text and prepared embeddings


# --- Keep DistilledDenoiser as is ---
class DistilledDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img")
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance")

        # Assuming distilled model doesn't need CFG or complex context
        txt_context = kwargs.pop("txt") # Get the single context prepared by `prepare`
        pass_through_kwargs = {k: v for k, v in kwargs.items() if k not in ['guidance', 'timesteps', 'img']}

        z = img
        guidance_vec = torch.full( (z.shape[0],), guidance, device=z.device, dtype=z.dtype ) # May not be used by model

        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full( (z.shape[0],), t_curr, dtype=z.dtype, device=z.device)
            pred = model(
                img=z,
                timesteps=t_vec,
                txt=txt_context, # Pass the context
                guidance=guidance_vec, # May be ignored
                **pass_through_kwargs
            )
            z = z + (t_prev - t_curr) * pred
        return z

    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Tensor]]:
        # Distilled typically uses only positive prompt
        model_t5 = optional_models.get("t5")
        context = model_t5(text).to(device, dtype)
        # No negative prompts needed for simple distilled sampling
        return text, {"txt": context} # Return context under 'txt' key assumed by `prepare`


# --- NEW PersonalizationDenoiser ---
class PersonalizationDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        z = kwargs.pop("img") # packed noisy latent z
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance") # Text guidance scale

        # Core Context Inputs (Prepared by self.prepare_guidance via prepare func)
        cross_attention_context = kwargs.pop("cross_attention_context")
        unconditional_context = kwargs.pop("unconditional_context")
        pass_through_kwargs = {k: v for k, v in kwargs.items() if k not in ['guidance', 'timesteps', 'img']}

        # Denoising Loop
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # Prepare input for this step: [conditional_latent, unconditional_latent]
            # Latent z is the same for both initially
            model_input = torch.cat([z, z], dim=0)
            t_vec = torch.full((model_input.shape[0],), t_curr, dtype=z.dtype, device=z.device)

            # Prepare combined context: [conditional_context, unconditional_context]
            # Assumes MMDiT's 'txt' argument takes the cross-attention context
            batched_context = torch.cat([cross_attention_context, unconditional_context], dim=0)
            model_kwargs = {**pass_through_kwargs, "txt": batched_context}

            # Duplicate other per-batch args (e.g., y_vec from CLIP text)
            for key in ['img_ids', 'txt_ids', 'y_vec']:
                 if key in model_kwargs:
                      key_val = model_kwargs[key]
                      # Ensure batch dim matches model_input
                      if key_val.shape[0] != model_input.shape[0]:
                          key_val = torch.cat([key_val, key_val], dim=0)
                      model_kwargs[key] = key_val

            # Forward pass through the model
            pred = model(
                img=model_input, # MMDiT expects 'img' key for the latent
                timesteps=t_vec,
                **model_kwargs # Pass the prepared context and other args
            )

            # Perform CFG (IP-Adapter style)
            cond_pred, uncond_pred = pred.chunk(2, dim=0)
            guided_pred = uncond_pred + guidance * (cond_pred - uncond_pred)
            # Note: personalization strength was applied during context prep

            # Euler step
            z = z + (t_prev - t_curr) * guided_pred

        return z # Return the denoised latent

    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        # --- NEW ARGUMENTS ---
        ref_image_paths: Optional[List[str]] = None,
        personalization_strength: float = 1.0,
        # --- End NEW ---
        **kwargs,
    ) -> Tuple[List[str], Dict[str, Tensor]]:
        """
        Prepares text embeddings, reference image embeddings, adds pseudo-embeddings,
        and combines them into conditional and unconditional contexts.
        """
        ret = {}
        model_clip = optional_models.get("clip")
        model_t5 = optional_models.get("t5")
        if model_clip is None or model_t5 is None:
            raise ValueError("CLIP and T5 models must be provided in optional_models for PersonalizationDenoiser")

        batch_size = len(text)
        clip_dim = model_clip.model.text_projection.shape[-1] # Get dim from model

        # --- Process Text ---
        text_tokens = model_t5(text).to(device, dtype) # T5 embeddings for conditional pass

        # --- Process Reference Images (if provided) ---
        augmented_ref_tokens_list = []
        valid_ref_provided = False
        if ref_image_paths:
            if len(ref_image_paths) != batch_size:
                 if len(ref_image_paths) == 1 and batch_size > 1:
                     ref_image_paths = ref_image_paths * batch_size
                 else:
                     raise ValueError(f"Number of ref paths ({len(ref_image_paths)}) must match batch size ({batch_size}) or be 1.")

            for i, ref_path in enumerate(ref_image_paths):
                if ref_path and os.path.exists(ref_path):
                    try:
                        ref_image = Image.open(ref_path).convert("RGB")
                        # --- GET CLIP IMAGE TOKENS ---
                        # THIS IS THE CRITICAL ASSUMPTION
                        if hasattr(model_clip, 'encode_image_tokens'):
                            ref_clip_tokens = model_clip.encode_image_tokens(ref_image) # Expected shape [1, NumRefTokens, Dim]
                            if ref_clip_tokens.ndim == 2: # Handle case where it might return pooled [1, Dim]
                                log_message("Warning: CLIP returned 2D tensor, repeating to simulate tokens.", level='warning')
                                ref_clip_tokens = ref_clip_tokens.unsqueeze(1).repeat(1, 77, 1) # Fallback
                        elif hasattr(model_clip.model, 'visual'): # Try accessing common vision transformer structure
                             # This depends heavily on the HFEmbedder impl for CLIP
                             vision_tower = model_clip.model.visual
                             # Assume embedder has preprocess
                             image_input = model_clip.preprocess(ref_image).to(device, dtype)
                             image_features = vision_tower(image_input.unsqueeze(0))[1] # Often [1, num_tokens, dim]
                             # May need projection if HFEmbedder doesn't do it
                             if hasattr(model_clip.model, 'visual_projection'):
                                 ref_clip_tokens = model_clip.model.visual_projection(image_features)
                             else:
                                 ref_clip_tokens = image_features # Use raw features if no projection layer found
                        else:
                            log_message("Warning: Cannot find 'encode_image_tokens' or 'model.visual'. Using pooled CLIP embedding as fallback.", level='warning')
                            pooled_emb = model_clip(ref_image).to(device, dtype) # Shape [1, Dim]
                            num_pseudo_tokens = 77
                            ref_clip_tokens = pooled_emb.unsqueeze(1).repeat(1, num_pseudo_tokens, 1)
                        # --- END GET CLIP IMAGE TOKENS ---

                        # Add pseudo concept embedding
                        pseudo_concept_emb = get_fixed_pseudo_concept_embedding(i, clip_dim, device, dtype)
                        augmented_ref_tokens = ref_clip_tokens.to(device, dtype) + pseudo_concept_emb * personalization_strength
                        augmented_ref_tokens_list.append(augmented_ref_tokens)
                        valid_ref_provided = True
                    except Exception as e:
                         log_message(f"Error processing reference image {ref_path} for batch item {i}: {e}", level='error')
                         augmented_ref_tokens_list.append(None) # Append None on error
                else:
                    augmented_ref_tokens_list.append(None) # Append None if path is invalid/empty

        # --- Combine Contexts ---
        # Pad reference tokens to max length within the batch if necessary
        final_ref_tokens_tensor = None
        if valid_ref_provided:
             max_ref_len = 0
             for tokens in augmented_ref_tokens_list:
                 if tokens is not None:
                     max_ref_len = max(max_ref_len, tokens.shape[1])

             if max_ref_len > 0:
                 padded_ref_tokens = []
                 for tokens in augmented_ref_tokens_list:
                     if tokens is not None:
                         pad_len = max_ref_len - tokens.shape[1]
                         padded = torch.cat([tokens, torch.zeros(1, pad_len, clip_dim, device=device, dtype=dtype)], dim=1) if pad_len > 0 else tokens
                     else:
                         padded = torch.zeros(1, max_ref_len, clip_dim, device=device, dtype=dtype)
                     padded_ref_tokens.append(padded)
                 final_ref_tokens_tensor = torch.cat(padded_ref_tokens, dim=0) # Shape [B, MaxRefLen, Dim]

        # Conditional context
        if final_ref_tokens_tensor is not None:
             # Ensure T5 and CLIP dims match - T5 output is usually the primary embedding dim for MMDiT
             if text_tokens.shape[-1] != final_ref_tokens_tensor.shape[-1]:
                 # Project CLIP tokens to T5 dim if necessary
                 # Requires adding a linear layer definition, maybe in optional_models? Or assume MMDiT handles projection.
                 # For now, let's assume MMDiT can handle concatenated contexts of potentially different dims or projects internally.
                 # This is another CRITICAL point depending on MMDiT architecture.
                 log_message(f"Warning: T5 dim ({text_tokens.shape[-1]}) and CLIP dim ({final_ref_tokens_tensor.shape[-1]}) mismatch. Concatenating directly.", level='warning')

             conditional_context = torch.cat([text_tokens, final_ref_tokens_tensor], dim=1)
        else:
             conditional_context = text_tokens
        ret["cross_attention_context"] = conditional_context

        # Unconditional context (Null text + augmented refs)
        null_text = [""] * batch_size
        null_text_tokens = model_t5(null_text).to(device, dtype)
        # Pad null text to match positive text length
        if null_text_tokens.shape[1] < text_tokens.shape[1]:
             pad_len = text_tokens.shape[1] - null_text_tokens.shape[1]
             null_text_tokens = torch.cat([null_text_tokens, torch.zeros(batch_size, pad_len, null_text_tokens.shape[-1], device=device, dtype=dtype)], dim=1)
        elif null_text_tokens.shape[1] > text_tokens.shape[1]:
             null_text_tokens = null_text_tokens[:, :text_tokens.shape[1], :]

        if final_ref_tokens_tensor is not None:
            # Concatenate null text with the same augmented & padded reference tokens
            unconditional_context = torch.cat([null_text_tokens, final_ref_tokens_tensor], dim=1)
        else:
            unconditional_context = null_text_tokens
        ret["unconditional_context"] = unconditional_context

        return text, ret


# Update Dictionary
SamplingMethodDict = {
    SamplingMethod.I2V: I2VDenoiser(),
    SamplingMethod.DISTILLED: DistilledDenoiser(),
    "personalize": PersonalizationDenoiser(), # Add specific key
}


# ======================================================
# Timesteps (Unchanged)
# ======================================================
def time_shift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1 + (alpha - 1) * t)

def get_res_lin_function(x1: float = 256, y1: float = 1, x2: float = 4096, y2: float = 3) -> callable:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(
    num_steps: int,
    image_seq_len: int, # Note: This might need adjustment based on latent shape/patching
    num_frames: int,
    shift_alpha: float | None = None,
    base_shift: float = 1,
    max_shift: float = 3,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        if shift_alpha is None:
            # Approximate image_seq_len if not directly known, e.g., from a typical latent size
            # spatial_latent_tokens = (height//D)*(width//D) # Requires height/width
            # image_seq_len = spatial_latent_tokens # Approximation
            shift_alpha = get_res_lin_function(y1=base_shift, y2=max_shift)(image_seq_len) # Use the passed arg
            shift_alpha *= math.sqrt(num_frames)
        timesteps = time_shift(shift_alpha, timesteps)
    return timesteps.tolist()


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    patch_size: int = 2,
    channel: int = 16, # This should be VAE latent channel size
) -> Tensor:
    """
    Generate noise in the VAE latent space shape BEFORE packing.
    """
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 8)) # VAE spatial compression
    T_vae = num_frames # Assuming no temporal compression in VAE here
    H_vae = math.ceil(height / D)
    W_vae = math.ceil(width / D)
    return torch.randn(
        num_samples,
        channel, # VAE channel size
        T_vae,
        H_vae,
        W_vae,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

# Keep pack/unpack, assuming patch_size=2 and VAE latent shapes are handled
def pack(x: Tensor, patch_size: int = 2) -> Tensor:
    # If patch_size is 1, packing should ideally do nothing or just reshape
    if patch_size == 1:
        b, c, t, h, w = x.shape
        return rearrange(x, 'b c t h w -> b (t h w) c')
    return rearrange(
        x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
    )

def unpack(
    x: Tensor, height: int, width: int, num_frames: int, patch_size: int = 2
) -> Tensor:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 8))
    H_vae = math.ceil(height / D)
    W_vae = math.ceil(width / D)
    T_vae = num_frames # Assuming no temporal compression

    # Handle patch_size=1 during unpack
    if patch_size == 1:
         b, seq, c = x.shape
         # Ensure seq matches T*H*W
         expected_seq = T_vae * H_vae * W_vae
         if seq != expected_seq:
              log_message(f"Warning: Unpack sequence length mismatch. Got {seq}, expected {expected_seq}. Reshaping may fail.", level="warning")
         return rearrange(x, 'b (t h w) c -> b c t h w', t=T_vae, h=H_vae, w=W_vae)

    # Original logic for patch_size > 1
    ph = pw = patch_size
    # Calculate packed height/width
    h_packed = math.ceil(H_vae / ph)
    w_packed = math.ceil(W_vae / pw)
    # Calculate expected channel dimension after packing
    c_packed = x.shape[-1] // (ph * pw)

    # Debugging shapes:
    # print(f"Unpack Input x shape: {x.shape}")
    # print(f"Target H_vae={H_vae}, W_vae={W_vae}, T_vae={T_vae}")
    # print(f"Using h_packed={h_packed}, w_packed={w_packed}, c_packed={c_packed}, ph={ph}, pw={pw}")

    try:
        return rearrange(
            x,
            "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
            h=h_packed,
            w=w_packed,
            t=T_vae,
            c=c_packed,
            ph=ph,
            pw=pw,
        )
    except Exception as e:
         log_message(f"Error during unpack rearrangement: {e}", level="error")
         log_message(f"Input shape: {x.shape}, Target dims: t={T_vae}, h={h_packed}, w={w_packed}, c={c_packed}, ph={ph}, pw={pw}", level="error")
         raise e


# ======================================================
# Prepare Function (Modified)
# ======================================================
def prepare(
    # Removed t5, clip args
    img: Tensor, # Packed noisy latent z, Shape: [B, SeqLen, Dim]
    prompt: list[str],
    optional_models: dict[str, nn.Module],
    # --- NEW ---
    ref_image_paths: Optional[List[str]] = None,
    personalization_strength: float = 1.0,
    # --- End NEW ---
    patch_size: int = 2, # Needed for IDs calculation
    # Add original shape info needed for IDs
    height: int = None,
    width: int = None,
    num_frames: int = None,
    neg: any = None,
    guidance_img: any = None,
    # Removed seq_align, T5 handles padding/alignment
) -> dict[str, Tensor]:
    """
    Prepare the input dict for the MMDiT model, including embeddings and IDs.
    Delegates context generation to the denoiser.
    """
    bs = img.shape[0]
    device, dtype = img.device, img.dtype

    is_personalization = ref_image_paths is not None and any(ref_image_paths)
    denoiser_key = "personalize" if is_personalization else SamplingMethod.I2V # Default to I2V if not personalizing

    # Get contexts using the appropriate denoiser's prepare_guidance
    denoiser = SamplingMethodDict[denoiser_key]
    text, additional_inp = denoiser.prepare_guidance(
        text=prompt,
        optional_models=optional_models,
        device=device,
        dtype=dtype,
        neg=None,
        guidance_img=kwargs.get("guidance_img", None) if 'kwargs' in locals() else None,
        ref_image_paths=ref_image_paths if is_personalization else None,
        personalization_strength=personalization_strength if is_personalization else 0.0,
    )

    # --- Prepare IDs ---
    # img_ids: Requires original unpacked latent dimensions
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 8))
    if height is None or width is None or num_frames is None:
        raise ValueError("Original height, width, and num_frames needed for ID generation.")
    T_vae = num_frames
    H_vae = math.ceil(height / D)
    W_vae = math.ceil(width / D)

    # Calculate IDs based on VAE latent shape *before* packing
    img_ids_unpacked = torch.zeros(T_vae, H_vae, W_vae, 3)
    img_ids_unpacked[..., 0] = torch.arange(T_vae)[:, None, None]
    img_ids_unpacked[..., 1] = torch.arange(H_vae)[None, :, None]
    img_ids_unpacked[..., 2] = torch.arange(W_vae)[None, None, :]
    # Instead of packing, flatten to sequence dimension to match packed latent
    img_ids = img_ids_unpacked.reshape(1, -1, 3)
    img_ids = img_ids.repeat(bs, 1, 1)


    # txt_ids: Placeholder based on the final context length
    # Note: The denoiser might return different contexts; use the primary one for shape
    context_key = "cross_attention_context" if is_personalization else "txt_cond"
    if context_key not in additional_inp:
         # Fallback if key is missing (e.g., DistilledDenoiser)
         context_key = list(additional_inp.keys())[0] # Get first available context key

    num_context_tokens = additional_inp[context_key].shape[1]
    txt_ids = torch.zeros(bs, num_context_tokens, 3, device=device, dtype=dtype)

    # --- Prepare y_vec (CLIP Text Embedding) ---
    model_clip = optional_models.get("clip")
    if model_clip is None: raise ValueError("CLIP model not found in optional_models")
    y_vec = model_clip(prompt).to(device, dtype)
    if y_vec.shape[0] == 1 and bs > 1:
        y_vec = repeat(y_vec, "1 ... -> bs ...", bs=bs)

    # Combine results into the final dictionary
    final_inp = {
        "img": img, # Packed noisy latent
        "img_ids": img_ids.to(device, dtype),
        "txt_ids": txt_ids.to(device, dtype),
        "y_vec": y_vec.to(device, dtype),
        **additional_inp # Add all prepared contexts
    }

    return final_inp


# prepare_ids is likely no longer needed
# def prepare_ids(...): pass


# ======================================================
# Prepare Models (Modified)
# ======================================================
# Keep the renamed original function
def _original_prepare_models(
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    offload_model: bool = False,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, dict[str, nn.Module]]:
    """Original model loading logic."""
    model_device = ("cpu" if offload_model else device)

    model = build_module(cfg.model, MODELS, device_map=model_device, torch_dtype=dtype).eval()
    model_ae = build_module(cfg.ae, MODELS, device_map=model_device, torch_dtype=dtype).eval()
    model_t5 = build_module(cfg.t5, MODELS, device_map=device, torch_dtype=dtype).eval()
    model_clip = build_module(cfg.clip, MODELS, device_map=device, torch_dtype=dtype).eval()
    if cfg.get("pretrained_lora_path", None) is not None:
        model = PeftModel.from_pretrained(model, cfg.pretrained_lora_path, is_trainable=False)

    optional_models = {}
    # Keep img_flux loading if needed for other parts
    if cfg.get("img_flux", None) is not None:
        model_img_flux = build_module(cfg.img_flux, MODELS, device_map=device, torch_dtype=dtype).eval()
        model_ae_img_flux = build_module(cfg.img_flux_ae, MODELS, device_map=device, torch_dtype=dtype).eval()
        optional_models["img_flux"] = model_img_flux
        optional_models["img_flux_ae"] = model_ae_img_flux

    return model, model_ae, model_t5, model_clip, optional_models

def prepare_models(
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    offload_model: bool = False,
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, dict[str, nn.Module]]:
    """
    Prepare models for inference and ensure T5/CLIP are in optional_models.
    """
    model, model_ae, model_t5, model_clip, optional_models = _original_prepare_models(cfg, device, dtype, offload_model)
    # Ensure T5 and CLIP are available for denoisers
    if 't5' not in optional_models: optional_models['t5'] = model_t5
    if 'clip' not in optional_models: optional_models['clip'] = model_clip
    return model, model_ae, model_t5, model_clip, optional_models


# ======================================================
# Prepare API (Modified)
# ======================================================
def prepare_api(
    model: nn.Module,
    model_ae: nn.Module,
    model_t5: nn.Module,
    model_clip: nn.Module,
    optional_models: dict[str, nn.Module],
    ref_image: str = None,
    noise_level: float = 0.0,
) -> callable:
    # Ensure T5/CLIP are in optional_models
    if 't5' not in optional_models: optional_models['t5'] = model_t5
    if 'clip' not in optional_models: optional_models['clip'] = model_clip

    @torch.inference_mode()
    def api_fn(
        opt: SamplingOption,
        # --- NEW API Args ---
        ref_image_paths: Optional[List[str]] = None,
        personalization_strength: Optional[float] = None,
        # --- End NEW ---
        cond_type: str = "t2v", # Used for standard I2V conditioning if no ref_image_paths
        seed: int = None,
        sigma_min: float = 1e-5, # Used for standard I2V
        text: list[str] = None,
        neg: list[str] = None,
        patch_size: int = 2, # Should ideally match model training
        # Removed channel arg, get from model or VAE
        **kwargs, # Other potential args
    ):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if seed is None:

            seed = opt.seed if opt.seed is not None else random.randint(0, 2**32 - 1)
        if opt.is_causal_vae:
            num_frames = (
                1
                if opt.num_frames == 1
                else (opt.num_frames - 1) // opt.temporal_reduction + 1
            )
        else:
            num_frames = (
                1 if opt.num_frames == 1 else opt.num_frames // opt.temporal_reduction
            )

        if ref_image is not None:

            image = load_image(ref_image, device=device, dtype=dtype, height=opt.height, width=opt.width, num_frames=opt.num_frames)

            vae_latent = model_ae.encode(image)
            if isinstance(vae_latent, (list, tuple)):
                vae_latent = vae_latent[0]
            z = personalize_latent(vae_latent, noise_level=noise_level)
            references = [(vae_latent[0, :, 0:1], vae_latent[0, :, -1:])] 
        else:
            z = get_noise(
                len(text),
                opt.height,
                opt.width,
                num_frames,
                device,
                dtype,
                seed,
                patch_size=patch_size,
                channel=channel // (patch_size**2),
            )
            references = [None] * len(text)

        denoiser = SamplingMethodDict[opt.method]

        timesteps = get_schedule(
            opt.num_steps,
            image_seq_len_approx,
            num_frames,
            shift=opt.shift,
            shift_alpha=opt.flow_shift,
        )

        text, additional_inp = denoiser.prepare_guidance(
            text=text,
            optional_models=optional_models,
            device=device,
            dtype=dtype,
            neg=neg,
            guidance_img=opt.guidance_img,
        )

        inp = prepare(model_t5, model_clip, z, prompt=text, patch_size=patch_size)
        inp.update(additional_inp)

        if opt.method in [SamplingMethod.I2V]:
            masks, masked_ref = prepare_inference_condition(
                z, cond_type, ref_list=references, causal=opt.is_causal_vae
            )
            inp["masks"] = masks
            inp["masked_ref"] = masked_ref
            inp["sigma_min"] = sigma_min

        x = denoiser.denoise(
            model,
            **inp,
            timesteps=timesteps,
            guidance=opt.guidance,
            text_osci=opt.text_osci,
            image_osci=opt.image_osci,
            scale_temporal_osci=(
                opt.scale_temporal_osci and "i2v" in cond_type
            ), 
            flow_shift=opt.flow_shift,
            patch_size=patch_size,
        )

        x = unpack(x, opt.height, opt.width, num_frames, patch_size=patch_size)

        # Handle standard I2V frame fixing if needed (and NOT personalizing)
        if not is_personalization and denoiser_key == SamplingMethod.I2V and references_i2v:
             # Frame fixing logic needs UNPACKED references
             unpacked_refs_i2v = []
             # This assumes collect_references_batch returns unpacked latents. If not, decode first.
             # For simplicity, assume they are VAE latents.
             for ref_batch_item in references_i2v:
                  if ref_batch_item is not None:
                       # ref_batch_item is list[Tensor], tensor shape [C, T', H_vae, W_vae]
                       unpacked_refs_i2v.append(ref_batch_item)
                  else:
                       unpacked_refs_i2v.append(None)


             for i in range(batch_size): # Iterate through batch
                 if unpacked_refs_i2v[i] is not None:
                     if cond_type == "i2v_head":
                         x[i, :, :1] = unpacked_refs_i2v[i][0][:, :1] # Use first ref, first frame
                     elif cond_type == "i2v_tail":
                         x[i, :, -1:] = unpacked_refs_i2v[i][-1][:, -1:] # Use last ref, last frame
                     elif cond_type == "i2v_loop":
                         x[i, :, :1] = unpacked_refs_i2v[i][0][:, :1] # Use first ref, first frame
                         x[i, :, -1:] = unpacked_refs_i2v[i][-1][:, -1:] # Use last ref, last frame


        # Decode final latent
        x = model_ae.decode(x)
        # Trim final output to the originally requested number of frames *before* temporal reduction
        x = x[:, :, : opt.num_frames]

        # Remove duplicate frames for standard I2V conditioning (if NOT causal VAE)
        if not is_personalization and not opt.is_causal_vae and denoiser_key == SamplingMethod.I2V:
            if hasattr(model_ae, 'compression') and isinstance(model_ae.compression, (list, tuple)) and len(model_ae.compression) > 0:
                pad_len = model_ae.compression[0] - 1 # Assuming temporal compression is first dim
                if pad_len > 0: # Only pad if temporal compression > 1
                    if cond_type == "i2v_head": x = x[:, :, pad_len:]
                    elif cond_type == "i2v_tail": x = x[:, :, :-pad_len]
                    elif cond_type == "i2v_loop": x = x[:, :, pad_len:-pad_len]


        return x

    return api_fn