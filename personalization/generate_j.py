import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from opensora.utils.config import Config
from opensora.utils.sampling import (
    SamplingOption, prepare_models, prepare_api, sanitize_sampling_option,
    prepare_ids, get_noise
)
from opensora.utils.inference import process_and_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
text_prompt = "a pink cat in the park"
ref_image_path = "/home/scur0552/personalization/assets/ref_images/cat.jpg"
output_dir = "/home/scur0552/personalization/outputs/personalized"
os.makedirs(os.path.join(output_dir, "video_256px"), exist_ok=True)

cfg = Config.fromfile("configs/diffusion/inference/256px.py")
model, model_ae, _, model_clip, optional_models = prepare_models(cfg, device, dtype)
api_fn = prepare_api(model, model_ae, None, model_clip, optional_models)

# Load CLIP model for joint text-image encoding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Try different image loading approaches
try:
    # First attempt: direct PIL
    image = Image.open(ref_image_path).convert("RGB")
except Exception as e:
    print(f"First attempt failed: {e}")
    try:
        # Second attempt: using imageio
        import imageio.v2 as imageio
        image = Image.fromarray(imageio.imread(ref_image_path)).convert("RGB")
    except Exception as e:
        print(f"Second attempt failed: {e}")
        try:
            # Third attempt: using opencv
            import cv2
            image = cv2.imread(ref_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        except Exception as e:
            print(f"All attempts failed. Please check the image file: {e}")
            raise

# Process text and image through CLIP
inputs = clip_processor(
    text=[text_prompt],
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    # Get joint embeddings from CLIP
    outputs = clip_model(**inputs)
    text_embedding = outputs.text_embeds
    image_embedding = outputs.image_embeds
    # Concatenate for joint conditioning
    joint_embedding = torch.cat([text_embedding, image_embedding], dim=-1)

cfg.save_dir = output_dir
cfg.seed = 42
cfg.dataset = {"data_path": "personalization"}
cfg.sampling_option["num_frames"] = 129
cfg.sampling_option["method"] = "distill"
cfg.sampling_option = sanitize_sampling_option(SamplingOption(**cfg.sampling_option))

z = get_noise(
    num_samples=1,
    height=cfg.sampling_option.height,
    width=cfg.sampling_option.width,
    num_frames=cfg.sampling_option.num_frames,
    device=device,
    dtype=dtype,
    seed=cfg.seed,
    patch_size=2,
    channel=cfg.model["in_channels"]
)

input_dict = prepare_ids(z, text_embedding=joint_embedding, clip_embedding=joint_embedding)

print("Generating personalized video with joint embedding")
try:
    with torch.no_grad():
        result = api_fn(
            cfg.sampling_option,
            seed=cfg.seed,
            patch_size=2,
            channel=cfg.model["in_channels"],
            text=[text_prompt],
            **input_dict
        )
        video = result.cpu()
        print("Video generated. Shape:", video.shape)
except Exception as e:
    import traceback
    print("Generation failed:")
    traceback.print_exc()
    exit(1)

process_and_save(
    video,
    {
        "text": [text_prompt],
        "name": ["joint_conditioning_video"]
    },
    cfg,
    "video_256px",
    cfg.sampling_option,
    0,
    0
)

print("Joint-conditioned video saved to:", output_dir)
