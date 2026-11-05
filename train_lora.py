"""Fine-tune Stable Diffusion 1.5 with LoRA.

Usage: python train_lora.py --arm ARM --tier 1000 --steps 400 [--seed 0 --exclude ids.csv --max-minutes 320]
Writes runs/<name>/lora.pt, loss.csv and config.json. With --max-minutes, training pauses (state.pt) and the same
command resumes it.
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common_sd import LATENTS, RUNS, SCALE, SPLIT, SD, encode_text, load_text, load_unet, lora_config, threads
from texts import ARMS, make_text

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=ARMS, required=True)
ap.add_argument("--tier", type=int, default=1000)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude", default="")
ap.add_argument("--name", default="")
ap.add_argument("--max-minutes", type=float, default=0)
args = ap.parse_args()
threads()
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
np_rng = np.random.default_rng(args.seed)

run = RUNS / (args.name or f"{args.arm}_t{args.tier}")
run.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SPLIT)
rows = df[(df.role == "gen_train") & (df.tier <= args.tier)].reset_index(drop=True)
if args.exclude:
    drop = set(pd.read_csv(args.exclude).ID)
    rows = rows[~rows.ID.isin(drop)].reset_index(drop=True)
lat_dir = LATENTS / str(args.res)
lat = np.stack([np.load(lat_dir / f"{i}.npy") for i in rows.ID]).astype(np.float32)
print(f"arm={args.arm} tier={args.tier} images={len(rows)} steps={args.steps} batch={args.batch}", flush=True)

from diffusers import DDPMScheduler
noise_sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler")
tok, enc = load_text()
unet = load_unet().requires_grad_(False)
unet.add_adapter(lora_config(args.rank))
params = [p for p in unet.parameters() if p.requires_grad]
print(f"trainable LoRA numbers: {sum(p.numel() for p in params) / 1e6:.2f} M "
      f"(of {sum(p.numel() for p in unet.parameters()) / 1e6:.0f} M in the U-Net)", flush=True)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
unet.train()

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
order, pos = np_rng.permutation(len(rows)), 0
log, t0, first = [], time.time(), 1
state = run / "state.pt"
if state.exists():
    saved = torch.load(state, weights_only=False)
    set_peft_model_state_dict(unet, saved["lora"])
    opt.load_state_dict(saved["opt"])
    rng.setstate(saved["rng"])
    np_rng.bit_generator.state = saved["np_rng"]
    torch.set_rng_state(saved["torch_rng"])
    order, pos, log, first = saved["order"], saved["pos"], saved["log"], saved["step"] + 1
    print(f"resumed at step {first}", flush=True)
for step in range(first, args.steps + 1):
    if pos + args.batch > len(order):
        order, pos = np_rng.permutation(len(rows)), 0
    idx = order[pos:pos + args.batch]
    pos += args.batch
    texts = [("" if rng.random() < args.drop else make_text(rows.iloc[i].to_dict(), args.arm, rng)) for i in idx]
    cond = encode_text(tok, enc, texts)
    m, s = torch.from_numpy(lat[idx, :4]), torch.from_numpy(lat[idx, 4:])
    x0 = (m + s * torch.randn_like(s)) * SCALE
    noise = torch.randn_like(x0)
    t = torch.randint(0, noise_sched.config.num_train_timesteps, (len(idx),))
    pred = unet(noise_sched.add_noise(x0, noise, t), t, cond).sample
    loss = F.mse_loss(pred.float(), noise.float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    log.append({"step": step, "loss": float(loss), "sec": round(time.time() - t0, 1)})
    if step % 25 == 0 or step == 1:
        recent = np.mean([r["loss"] for r in log[-25:]])
        print(f"step {step}/{args.steps}  loss {recent:.4f}  {time.time() - t0:.0f} s", flush=True)
    if args.max_minutes and step < args.steps and time.time() - t0 > 60 * args.max_minutes:
        torch.save({"lora": get_peft_model_state_dict(unet), "opt": opt.state_dict(), "rng": rng.getstate(),
                    "np_rng": np_rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "order": order,
                    "pos": pos, "log": log, "step": step}, state)
        print(f"paused at step {step}", flush=True)
        raise SystemExit(0)

torch.save(get_peft_model_state_dict(unet), run / "lora.pt")
state.unlink(missing_ok=True)
pd.DataFrame(log).to_csv(run / "loss.csv", index=False)
cfg = vars(args) | {"images": len(rows), "seconds": round(time.time() - t0)}
(run / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"saved {run} in {time.time() - t0:.0f} s", flush=True)
