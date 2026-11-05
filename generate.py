"""Generate one image per test item with a trained LoRA (or the base model with --run base).

Usage: python generate.py --run NAME [--arm ARM --ids items.csv --shard 0/4 --out NAME2 --steps 15 --cfg 4.0]
Writes runs/<out>/gen/<ID>.png and runs/<out>/prompts.csv (prompts_<i>.csv for a shard).
"""
import argparse
import json
import time

import pandas as pd
import torch

from common_sd import RUNS, SD, SPLIT, load_unet, load_vae, lora_config, threads
from texts import make_text

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--arm", default=None)
ap.add_argument("--n-per-mod", type=int, default=50)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--ids", default="")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default="")
args = ap.parse_args()
threads()

run = RUNS / args.run
run.mkdir(parents=True, exist_ok=True)
arm = args.arm or (json.loads((run / "config.json").read_text())["arm"] if args.run != "base" else "minim_paper")
out_run = RUNS / (args.out or args.run)
out = out_run / "gen"
out.mkdir(parents=True, exist_ok=True)

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
unet = load_unet()
if args.run != "base":
    from peft.utils import set_peft_model_state_dict
    cfg = json.loads((run / "config.json").read_text())
    unet.add_adapter(lora_config(cfg["rank"]))
    set_peft_model_state_dict(unet, torch.load(run / "lora.pt"))
pipe = StableDiffusionPipeline.from_pretrained(SD, unet=unet, vae=load_vae(), safety_checker=None,
                                               requires_safety_checker=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

df = pd.read_csv(SPLIT)
test = df[df.role == "test"]
if args.ids:
    test = test[test.ID.isin(set(pd.read_csv(args.ids).ID))]
else:
    test = test.groupby("modality", group_keys=False).head(args.n_per_mod)
part, parts = map(int, args.shard.split("/"))
test = test.reset_index(drop=True)
test = test.iloc[part * len(test) // parts:(part + 1) * len(test) // parts].reset_index(drop=True)
prompts = []
t0 = time.time()
for k, row in test.iterrows():
    path = out / f"{row.ID}.png"
    text = make_text(row.to_dict(), arm, for_eval=True)
    prompts.append({"ID": row.ID, "modality": row.modality, "region": row.region, "prompt": text})
    if path.exists():
        continue
    seed = 1000 + int(row.ID.split("_")[-1])
    img = pipe(text, num_inference_steps=args.steps, guidance_scale=args.cfg, height=args.res, width=args.res,
               generator=torch.Generator().manual_seed(seed)).images[0]
    img.save(path)
    if k % 10 == 0:
        print(f"{k + 1}/{len(test)}  {time.time() - t0:.0f} s", flush=True)
pd.DataFrame(prompts).to_csv(out_run / ("prompts.csv" if parts == 1 else f"prompts_{part}.csv"), index=False)
print(f"done: {len(test)} images in {time.time() - t0:.0f} s -> {out}", flush=True)
