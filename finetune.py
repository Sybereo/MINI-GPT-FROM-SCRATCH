"""
Supervised Fine-Tuning (SFT) script for custom Transformer LLM.
- Multi-turn & single-turn instruction following.
- Proper loss masking (-1 for model.py ignore_index).
- Safe embedding table expansion (32000 -> 32064).
"""
import dataclasses
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

from model import ProfessionalLLM
from tokenizer.bpe import BPETokenizer
from train import configure_optimizers, get_lr

CFG = dict(
    sources=[
        # Main instruction data
        dict(path="alpaca_data.json", max_examples=10000, val_size=400),
        # Keep a little of your casual chat style
        dict(path="chit_chat_200.json", max_examples=200, val_size=20),
    ],
    base_ckpt="checkpoints/latest.pt",
    tokenizer_in="tokenizer/owt_pro.json",
    tokenizer_out="tokenizer/owt_reasoning.json",
    out_dir="checkpoints",
    max_len=512,
    batch_size=8,
    grad_accum=4,
    epochs=3,              # increased from 2
    lr=5e-5,               # lowered from 1e-4 (important)
    min_lr=1e-5,
    warmup_steps=100,      # a bit more warmup
    weight_decay=0.05,     # slightly higher
    grad_clip=1.0,
    seed=42,
)
DEMO_PROMPTS = [
    "Hi!",
    "What is the capital of France?",
    "Explain why the sky is blue in simple terms.",
]

def row_to_messages(row):
    if "messages" in row:
        raw = [(m.get("role"), (m.get("content") or "").strip()) for m in row["messages"]]
    elif "instruction" in row:
        instruction = (row.get("instruction") or "").strip()
        extra = (row.get("input") or "").strip()
        raw = [
            ("user", instruction + ("\n\n" + extra if extra else "")),
            ("assistant", (row.get("output") or "").strip())
        ]
    else:
        return []
    
    clean = []
    for role, text in raw:
        if role not in ("user", "assistant"):
            continue
        if not text or role != ("user" if len(clean) % 2 == 0 else "assistant"):
            break
        clean.append((role, text))
    if len(clean) % 2 == 1:
        clean = clean[:-1]
    return clean

def build_example(tok, messages, max_len):
    user = tok.special_tokens["<|user|>"]
    assistant = tok.special_tokens["<|assistant|>"]
    eos = tok.special_tokens["<|endoftext|>"]

    ids, is_target = [], []
    for i in range(0, len(messages) - 1, 2):
        prompt = [user] + tok.encode(messages[i][1]) + [assistant]
        reply = tok.encode(messages[i + 1][1]) + [eos]

        if len(ids) + len(prompt) + len(reply) > max_len + 1:
            break
        ids += prompt + reply
        is_target += [False] * len(prompt) + [True] * len(reply)

    if not ids:
        return None

    x = ids[:-1]
    # Use -1 to match ignore_index=-1 inside model.py
    y = [ids[j + 1] if is_target[j + 1] else -1 for j in range(len(ids) - 1)]
    return x, y

def split_examples(examples, val_size):
    n_val = min(val_size, max(1, len(examples) // 10))
    return examples[:n_val], examples[n_val:]

def load_sources(sources, tok, max_len, seed):
    rng = random.Random(seed)
    train, val = [], []
    for s in sources:
        if not os.path.exists(s["path"]):
            print(f"WARNING: {s['path']} not found - skipping.")
            continue
        with open(s["path"], "r", encoding="utf-8") as f:
            rows = json.load(f)
        rng.shuffle(rows)
        if s.get("max_examples"):
            rows = rows[: s["max_examples"]]

        examples = []
        for row in rows:
            ex = build_example(tok, row_to_messages(row), max_len)
            if ex is not None:
                examples.append(ex)

        v, t = split_examples(examples, s.get("val_size", 200))
        repeat = s.get("repeat", 1)
        train += t * repeat
        val += v
        print(f"{s['path']}: loaded {len(examples):,} valid examples ({len(t):,} train, {len(v):,} val).")
    return train, val

def make_batches(examples, batch_size, shuffle, rng):
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [examples[j] for j in idx[i : i + batch_size]]

def collate(batch, pad_id, device):
    length = max(len(x) for x, _ in batch)
    xs = torch.full((len(batch), length), pad_id, dtype=torch.long)
    ys = torch.full((len(batch), length), -1, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        ys[i, : len(y)] = torch.tensor(y, dtype=torch.long)
    return xs.to(device), ys.to(device)

def load_base_model_resized(ckpt_path, new_vocab, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    old_sd = ckpt["model_state_dict"]
    old_w = old_sd["tok_embeddings.weight"]
    old_vocab = old_w.shape[0]

    if new_vocab < old_vocab:
        raise ValueError(f"New vocab ({new_vocab}) is smaller than checkpoint vocab ({old_vocab})")

    args = dataclasses.replace(ckpt["args"], vocab_size=new_vocab)
    model = ProfessionalLLM(args).to(device)

    rest = {k: v for k, v in old_sd.items() if k not in ("tok_embeddings.weight", "output.weight")}
    model.load_state_dict(rest, strict=False)

    with torch.no_grad():
        w = model.tok_embeddings.weight
        w[:old_vocab] = old_w
        if new_vocab > old_vocab:
            noise = 0.02 * torch.randn(new_vocab - old_vocab, old_w.shape[1], device=device)
            w[old_vocab:] = old_w.mean(0, keepdim=True) + noise

    if hasattr(model, "output") and model.output.weight is not model.tok_embeddings.weight:
        with torch.no_grad():
            model.output.weight[:old_vocab] = old_w
            if new_vocab > old_vocab:
                model.output.weight[old_vocab:] = old_w.mean(0, keepdim=True)

    print(f"Resized embedding table: {old_vocab} -> {new_vocab} (kept trained rows)")
    return model, args, ckpt.get("iter_num", 0)

@torch.no_grad()
def generate_reply(model, tok, user_text, device="cuda", max_new_tokens=100):
    model.eval()
    user = tok.special_tokens["<|user|>"]
    assistant = tok.special_tokens["<|assistant|>"]
    eos = tok.special_tokens["<|endoftext|>"]

    prompt_ids = [user] + tok.encode(user_text) + [assistant]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    out = []
    for _ in range(max_new_tokens):
        if x.size(1) >= model.args.max_seq_len:
            break
        logits, _ = model(x)
        logits = logits[0, -1].float()
        nxt = int(torch.argmax(logits))
        if nxt == eos:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)

    return tok.decode(out).strip()

def demo(model, tok, device):
    for prompt in DEMO_PROMPTS:
        reply = generate_reply(model, tok, prompt, device=device)
        print(f"  User: {prompt}\n  Model: {reply}\n")

@torch.no_grad()
def evaluate(model, examples, cfg, pad_id, device, amp_dtype):
    model.eval()
    total, count = 0.0, 0
    for batch in make_batches(examples, cfg["batch_size"], False, None):
        x, y = collate(batch, pad_id, device)
        with torch.autocast(device_type=device, dtype=amp_dtype):
            _, loss = model(x, targets=y)
        k = int((y != -1).sum())
        total += loss.item() * k
        count += k
    model.train()
    return total / max(count, 1)

def save_checkpoint(path, model, args, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model_state_dict": model.state_dict(), "args": args, **extra}, tmp)
    os.replace(tmp, path)

def main():
    cfg = CFG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    print(f"--- Running Conversational SFT on {device.upper()} ---")

    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    tok = BPETokenizer.load(cfg["tokenizer_in"])
    tok.add_special_tokens(["<|user|>", "<|assistant|>", "<think>", "</think>"])
    tok.save(cfg["tokenizer_out"])

    new_vocab = ((tok.n_vocab + 63) // 64) * 64
    print(f"Registered tokens: {tok.special_tokens}")
    print(f"Resizing target vocab size to: {new_vocab}")
    pad_id = tok.special_tokens.get("<|pad|>", 0)

    train, val = load_sources(cfg["sources"], tok, cfg["max_len"], cfg["seed"])
    if len(train) < 50 or len(val) < 1:
        raise SystemExit("Dataset too small. Verify alpaca_data.json exists and is valid.")

    print(f"Training on {len(train):,} samples | Validating on {len(val):,} samples")

    model, args, _ = load_base_model_resized(cfg["base_ckpt"], new_vocab, device)
    model.train()

    optimizer = configure_optimizers(model, cfg["weight_decay"], cfg["lr"], device)
    scaler = torch.amp.GradScaler(device=device, enabled=use_fp16)

    micro_per_epoch = math.ceil(len(train) / cfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = dict(warmup_iters=cfg["warmup_steps"], total_iters=total_steps, lr=cfg["lr"], min_lr=cfg["min_lr"])

    step = 0
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        batches = list(make_batches(train, cfg["batch_size"], True, rng))
        run_loss, run_n = 0.0, 0
        for b_i, batch in enumerate(batches):
            x, y = collate(batch, pad_id, device)
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(x, targets=y)
            run_loss += loss.item()
            run_n += 1
            scaler.scale(loss / cfg["grad_accum"]).backward()

            if (b_i + 1) % cfg["grad_accum"] == 0 or b_i == len(batches) - 1:
                lr = get_lr(step, sched)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 50 == 0:
                    print(f"Epoch {epoch+1}/{cfg['epochs']} | Step {step:4d}/{total_steps} | "
                          f"Loss: {run_loss / run_n:.4f} | LR: {lr:.2e} | Time: {time.time() - t0:.0f}s")
                    run_loss, run_n = 0.0, 0

        val_loss = evaluate(model, val, cfg, pad_id, device, amp_dtype)
        print(f"\n=== Epoch {epoch + 1} Complete | Val Loss: {val_loss:.4f} ===")
        demo(model, tok, device)
        model.train()

    save_checkpoint(os.path.join(cfg["out_dir"], "sft_final.pt"), model, args, epoch=cfg["epochs"])
    print(f"Saved SFT checkpoint to {os.path.join(cfg['out_dir'], 'sft_final.pt')}")
    print(f"Saved reasoning tokenizer to {cfg['tokenizer_out']}")

if __name__ == "__main__":
    main()
