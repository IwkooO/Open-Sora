import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from opensora.utils.config import Config
from opensora.utils.sampling import (
    SamplingOption,
    prepare_models,
    prepare_api,
    sanitize_sampling_option
)
from opensora.utils.inference import process_and_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_prompt = "a dog in the garden"  
ref_image_path = "/home/scur0552/personalization/assets/ref_images/rottweiler.jpg"
output_dir = "/home/scur0552/personalization/outputs/personalized"
os.makedirs(os.path.join(output_dir, "video_256px"), exist_ok=True)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

cfg = Config.fromfile("configs/diffusion/inference/256px.py")
model, model_ae, model_t5, model_clip, optional_models = prepare_models(cfg, device, torch.bfloat16)
t5_encoder = model_t5.hf_module
t5_tokenizer = model_t5.tokenizer

image = Image.open(ref_image_path).convert("RGB")
image_inputs = clip_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    vision_output = clip_model.vision_model(**image_inputs).last_hidden_state 
    image_tokens = vision_output[:, 1:, :]  

proj_layer = torch.nn.Linear(1024, 4096).to(device)
image_tokens_proj = proj_layer(image_tokens)  

text_inputs = t5_tokenizer(
    text_prompt,
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True
).to(device)

with torch.no_grad():
    text_embedding = t5_encoder(**text_inputs).last_hidden_state  

combined_embedding = torch.cat([text_embedding, image_tokens_proj], dim=1)  

cfg.save_dir = output_dir
cfg.seed = 42
cfg.sampling_option["num_frames"] = 129
cfg.sampling_option = sanitize_sampling_option(SamplingOption(**cfg.sampling_option))
cfg.dataset = {"data_path": "personalization"} 

api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

print("Generating personalized video...")
try:
    with torch.no_grad():
        result = api_fn(
            cfg.sampling_option,
            "t2v",
            seed=cfg.seed,
            patch_size=2,
            channel=cfg.model["in_channels"],
            text=[text_prompt],          
            context=combined_embedding   
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
        "name": ["personalized_video"]
    },
    cfg,
    "video_256px",
    cfg.sampling_option,
    0,
    0
)

print("Personalized video saved to:", output_dir)
