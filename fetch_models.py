"""Download the pinned model files into the Hugging Face cache."""
from pathlib import Path

from huggingface_hub import constants, snapshot_download

PINS = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": ("451f4fe16113bff5a5d2269ed5ad43b0592e9a14", [
        "model_index.json", "feature_extractor/*", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
        "text_encoder/model.safetensors", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json"]),
    "stabilityai/sd-vae-ft-mse": ("31f26fdeee1355a5c34592e401dd41e45d25a493", [
        "config.json", "diffusion_pytorch_model.safetensors"]),
}

for repo, (commit, files) in PINS.items():
    snapshot_download(repo, revision=commit, allow_patterns=files)
    ref = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit)
    print(repo, commit)
