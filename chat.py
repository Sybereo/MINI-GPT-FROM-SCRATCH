import os
import torch
import torch.nn.functional as F
from tokenizer.bpe import BPETokenizer
from model import ProfessionalLLM, ModelArgs

def sample_top_k_top_p(logits, top_k=30, top_p=0.85):
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        kth_val = v[..., -1, None]
        logits = torch.where(logits < kth_val, torch.tensor(-float('inf'), device=logits.device), logits)

    probs = F.softmax(logits, dim=-1)

    if top_p > 0.0:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token_sorted = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token_sorted)
    else:
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def chat_generate(model, tokenizer, prompt, max_new_tokens=250, temperature=0.6, top_k=40, top_p=0.9, repetition_penalty=1.15, device="cuda"):
    model.eval()
    
    user_token = tokenizer.special_tokens.get("<|user|>", -1)
    assistant_token = tokenizer.special_tokens.get("<|assistant|>", -1)
    eos_token = tokenizer.special_tokens.get("<|endoftext|>", -1)

    # Strictly matches the SFT training format: <|user|> prompt <|assistant|>
    prompt_ids = [user_token] + tokenizer.encode(prompt) + [assistant_token]
    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    max_ctx = getattr(model.args, "max_seq_len", 512)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) >= max_ctx:
                break

            logits, _ = model(tokens)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty across generated tokens
            for token_id in set(tokens[0].tolist()):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

            next_token = sample_top_k_top_p(next_token_logits, top_k=top_k, top_p=top_p)
            token_id = next_token.item()
            
            # Halting tokens - stop generating immediately if we hit endoftext or user token
            if token_id in (eos_token, user_token):
                break

            token_str = tokenizer.decode([token_id])
            print(token_str, end="", flush=True)
            
            tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)

    print("\n" + "-"*50)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tok_path = "tokenizer/owt_reasoning.json"
    if not os.path.exists(tok_path):
        tok_path = "tokenizer/owt_chat.json"

    tokenizer = BPETokenizer.load(tok_path)
    
    ckpt_path = "checkpoints/sft_final.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Please train with finetune.py first.")
        return

    print(f"Loading checkpoint: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "args" in checkpoint:
        args = checkpoint["args"]
    else:
        vocab_size = checkpoint.get("vocab_size", 32064)
        args = ModelArgs(dim=512, n_layers=6, n_heads=8, vocab_size=vocab_size, max_seq_len=512)

    model = ProfessionalLLM(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n=== Model Online ===")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit']:
            break

        print("Model:")
        chat_generate(model, tokenizer, user_input, max_new_tokens=250, device=device)

if __name__ == "__main__":
    main()
