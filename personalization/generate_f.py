import os
import torch
from opensora.utils.config import Config
from opensora.utils.sampling import (
    SamplingOption, prepare_models, prepare_api, sanitize_sampling_option
)
from opensora.utils.inference import process_and_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_prompt = "a cat in the beach. keep the cat's face in the frame and the cat's body in the frame. ensure the cat is in the frame for the entire video and the background is the beach."
ref_image_path = "/home/pnair/Open-Sora/personalization/assets/ref_images/cat.jpg"
output_dir = "/home/pnair/Open-Sora/personalization/outputs/personalized/video_256px"
os.makedirs(os.path.join(output_dir, "video_256px"), exist_ok=True)

cfg = Config.fromfile("configs/diffusion/inference/256px.py")
model, model_ae, model_t5, model_clip, optional_models = prepare_models(cfg, device, torch.bfloat16)
api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional_models)

cfg.save_dir = output_dir
cfg.seed = 42
cfg.dataset = {"data_path": "personalization"}
cfg.sampling_option["num_frames"] = 129
cfg.sampling_option["method"] = "i2v"
cfg.sampling_option = sanitize_sampling_option(SamplingOption(**cfg.sampling_option))

print("Generating personalized video using i2v_head...")
try:
    with torch.no_grad():
        result = api_fn(
            cfg.sampling_option,
            cond_type="i2v_head",
            seed=cfg.seed,
            patch_size=2,
            channel=cfg.model["in_channels"],
            text=[text_prompt],
            ref=[ref_image_path]  
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
