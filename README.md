# diff-test

GitHub Actions jobs that fine-tune Stable Diffusion 1.5 with LoRA on CPU and generate test images.

- `train_lora.py`, `generate.py`, `common_sd.py`: training and generation.
- `plans/`: the jobs a run starts (`full.json`, `smoke.json`); `plan.py` turns a plan into job lists.
- `locked/`: inputs (data subset, text builder, earlier LoRA weights), encrypted. The key is a repository secret, and
  outputs are encrypted the same way before upload.
- `preregistration.sha256`: SHA-256 of the claims file, committed before the first run.

Run: Actions → runs → Run workflow, with a plan file.

Data: a subset of ROCOv2 (Rückert et al., 2024), CC BY-NC 4.0. Models: Stable Diffusion 1.5 and sd-vae-ft-mse from
Hugging Face, pinned to fixed revisions in `fetch_models.py`.
