"""
Pretraining loop (resume-safe, cosine LR, multi-file data).
"""
import glob
import math
import os
import re
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import ProfessionalLLM, ModelArgs

CFG = dict(
    bin_paths=["data/owt_3gb.bin"],     # add more .bin files here as you encode more text
    local_cache="/content/data_cache",  # fast local disk; set to None to read in place
    checkpoint_dir="checkpoints",
    seq_len=512,
    batch_size=16,
    grad_accum=4,
    total_iters=16000,                  # total optimizer steps (cosine schedule ends here)
    lr=6e-4,
    min_lr=6e-5,
    warmup_iters=100,
    weight_decay=0.1,
    grad_clip=1.0,
    # model shape: must match the checkpoint you are resuming from
    dim=512,
    n_layers=6,
    n_heads=8,
    vocab_size=32000,
    latest_every=500,
    milestone_every=5000,
    log_every=10,
    num_workers=2,
)


class MemmapDataset(Dataset):
    """Non-overlapping (x, y) windows over one or more uint16 token files."""

    def __init__(self, paths, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.files = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.total_tokens = sum(len(f) for f in self.files)
        windows = [(len(f) - 1) // seq_len for f in self.files]
        self.offsets = np.cumsum([0] + windows)

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        f = int(np.searchsorted(self.offsets, idx, side="right")) - 1
        start = (idx - int(self.offsets[f])) * self.seq_len
        chunk = torch.from_numpy(self.files[f][start : start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def cache_locally(paths, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for p in paths:
        dst = os.path.join(cache_dir, os.path.basename(p))
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(p):
            print(f"Copying {p} -> {dst} ...")
            shutil.copyfile(p, dst)
        out.append(dst)
    return out


def configure_optimizers(model, weight_decay, learning_rate, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=(device_type == "cuda")
    )


def get_lr(it, cfg):
    """Linear warmup, then cosine decay from lr to min_lr. `it` is 0-based."""
    if it < cfg["warmup_iters"]:
        return cfg["lr"] * (it + 1) / cfg["warmup_iters"]
    if it >= cfg["total_iters"]:
        return cfg["min_lr"]
    ratio = (it - cfg["warmup_iters"]) / (cfg["total_iters"] - cfg["warmup_iters"])
    return cfg["min_lr"] + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg["lr"] - cfg["min_lr"])


def find_latest_checkpoint(ckpt_dir):
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    best = None
    for p in glob.glob(os.path.join(ckpt_dir, "model_step_*.pt")):
        m = re.search(r"model_step_(\d+)\.pt$", p)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best[1] if best else None


def save_checkpoint(path, model, optimizer, scaler, iter_num, loss, args):
    tmp = path + ".tmp"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "iter_num": iter_num,
            "loss": loss,
            "args": args,
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic


def main(overrides=None):
    cfg = {**CFG, **(overrides or {})}
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    print(f"--- Starting Optimization Engine on {device.upper()} ---")

    paths = cfg["bin_paths"]
    if cfg["local_cache"]:
        paths = cache_locally(paths, cfg["local_cache"])
    dataset = MemmapDataset(paths, cfg["seq_len"])
    print(f"Dataset: {dataset.total_tokens:,} tokens in {len(paths)} file(s)")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
        pin_memory=use_fp16,
        persistent_workers=cfg["num_workers"] > 0,
    )
    data_iter = iter(dataloader)

    args = ModelArgs(
        dim=cfg["dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["seq_len"],
    )
    model = ProfessionalLLM(args).to(device)
    optimizer = configure_optimizers(model, cfg["weight_decay"], cfg["lr"], device)
    scaler = torch.amp.GradScaler(device=device, enabled=use_fp16)

    start_iter = 0
    ckpt_path = find_latest_checkpoint(cfg["checkpoint_dir"])
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_fp16 and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_iter = ckpt["iter_num"]
        del ckpt
        print(f"Resumed from {ckpt_path} at step {start_iter}")
    if start_iter >= cfg["total_iters"]:
        print(f"Already at step {start_iter} >= total_iters {cfg['total_iters']}; nothing to do.")
        return

    accum = cfg["grad_accum"]
    tokens_per_step = cfg["batch_size"] * cfg["seq_len"] * accum
    print(f"{tokens_per_step:,} tokens/step, {cfg['total_iters'] - start_iter:,} steps to go")

    model.train()
    t0 = time.time()
    data_t = 0.0
    iter_num = start_iter
    loss_accum = torch.zeros((), device=device)

    for iter_num in range(start_iter + 1, cfg["total_iters"] + 1):
        lr = get_lr(iter_num - 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        for _ in range(accum):
            td = time.time()
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            data_t += time.time() - td

            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(x, targets=y)
                loss = loss / accum

            loss_accum += loss.detach()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if iter_num % cfg["log_every"] == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * cfg["log_every"] / dt
            print(
                f"Step {iter_num:5d} | Loss: {loss_accum.item():.4f} | LR: {lr:.2e} | "
                f"Tok/sec: {tps:,.0f} | Data wait: {100 * data_t / dt:.0f}%"
            )
            data_t = 0.0

        if iter_num % cfg["latest_every"] == 0:
            save_checkpoint(
                os.path.join(cfg["checkpoint_dir"], "latest.pt"),
                model, optimizer, scaler, iter_num, loss_accum.item(), args,
            )
        if iter_num % cfg["milestone_every"] == 0:
            save_checkpoint(
                os.path.join(cfg["checkpoint_dir"], f"model_step_{iter_num}.pt"),
                model, optimizer, scaler, iter_num, loss_accum.item(), args,
            )

    save_checkpoint(
        os.path.join(cfg["checkpoint_dir"], "latest.pt"),
        model, optimizer, scaler, iter_num, loss_accum.item(), args,
    )
    print(f"Done at step {iter_num}.")


if __name__ == "__main__":
    main()
