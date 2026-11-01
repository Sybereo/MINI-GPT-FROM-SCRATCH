# A 37M-parameter GPT-style chat model, built from scratch

A small decoder-only language model with its own byte-level BPE tokenizer, trained end to end on a single free Colab T4: pretraining on web text, then supervised fine-tuning (SFT) so it can hold a short conversation.

PyTorch supplies tensors and autograd. Everything else is written by hand: the tokenizer, the model, the training loop, the fine-tuning pipeline and the sampler. The model is deliberately small. It is a learning project and a working demo, not a competitor to anything.

*What to expect: replies are fluent and usually on topic, but often factually wrong. See Limitations.*

## Quickstart

You need `tokenizer/owt_reasoning.json` and `checkpoints/sft_final.pt`.
**[📥 Download weights/sft_final.pt from Releases]([https://github.com/Sybereo/MINI-GPT-FROM-SCRATCH/releases](https://github.com/Sybereo/MINI-GPT-FROM-SCRATCH/releases/download/v1/sft_final.pt))**

```bash
pip install torch regex numpy
python chat.py
```

Type `quit` or `exit` to stop. It runs on CPU or GPU. Each reply is generated token by token without a KV cache, which is fine at this size but not fast.

## Model

| Feature | Specification |
| :--- | :--- |
| **Type** | decoder-only transformer (Llama-style) |
| **Layers / width / heads** | 6 / 512 / 8 (head dim 64) |
| **Feed-forward** | SwiGLU, hidden size 1536 |
| **Normalisation** | RMSNorm, pre-norm |
| **Positions** | rotary embeddings (RoPE) |
| **Context length** | 512 tokens |
| **Embeddings** | input and output tied |
| **Vocabulary** | 32,064 rows (32,000 base + special tokens, padded to multiple of 64) |
| **Parameters** | about 36.9M total, 20.5M excluding the embedding table |

The architecture follows Llama's design choices (RMSNorm, RoPE, SwiGLU). The tokenizer's pre-splitting regex is the one used by GPT-4 (cl100k).

## Tokenizer

Byte-level BPE with a 32,000-token base vocabulary (256 bytes plus 31,744 merges), trained on the first 500 MB of OpenWebText. Byte-level means any string can be encoded, so there is no unknown token. It averages about 4.5 bytes per token on OpenWebText.

Special tokens added after training: `<|endoftext|>`, `<|pad|>`, `<|mask|>`, `<|user|>`, `<|assistant|>`. Special-token strings typed by a user are treated as plain text, not as control tokens.

## Training

### Pretraining
* **Data:** a subset of OpenWebText, 3 GB of text → 699M tokens.
* **Schedule:** 16,000 steps of 32,768 tokens each (~524M tokens seen). The first 5,000 steps used only the first 500 MB; the remaining 11,000 steps used the full 3 GB.
* **Optimizer:** AdamW, mixed precision (fp16 with loss scaling), gradient clipping at 1.0. Learning rate started at 6e-4, then warmup + cosine decay down to 6e-5.
* **Hardware:** one Colab T4, ~40,000 tokens per second.

| Step | Training loss |
| :--- | :--- |
| 10 | 8.1 |
| 5,000 | ~4.3 |
| 16,000 | ~3.9 (perplexity roughly 50) |

*These are single-batch training losses, so they are noisy. No pretraining validation set was held out.*

### Supervised fine-tuning
* **Data:** Stanford Alpaca + a small set of hand-written casual conversations.
* **Format:** `<|user|>` … `<|assistant|>` … `<|endoftext|>`, repeated for each turn. Long conversations keep as many whole turns as fit in 512 tokens.
* **Loss masking:** the loss covers only assistant text and the closing `<|endoftext|>`. Everything else is set to the ignore label (`-1`), so the model learns to answer and to stop, and does not learn to imitate the user.
* **Embedding resize:** the pretrained table had 32,000 rows. It was expanded to 32,064 rows; the original 32,000 rows were kept unchanged and the new rows were initialised from the mean embedding plus small noise.

## What went wrong, and what it taught me

* **Special tokens outside the embedding table:** I registered `<|endoftext|>` and the others after training the 32,000-token vocabulary, so their ids were ≥ 32,000. The pretrained model only had rows 0–31,999. The model could never emit `<|endoftext|>` and therefore never learned to stop.
  * *Fix:* resize the embedding table before fine-tuning.
* **Label masking value:** PyTorch’s default `ignore_index` is `-100`, but the model’s loss used `-1`. Labels masked with the wrong value were treated as real class ids and crashed the GPU with a device-side assert.
  * *Fix:* use one consistent masking value everywhere.
* **Model-size mismatch at load time:** The chat script built the model from the tokenizer’s reported size while the checkpoint had 32,064 rows.
  * *Fix:* always rebuild the model from the configuration stored inside the checkpoint.
* **Repetition loops:** A small model easily gets stuck repeating a phrase.
  * *Mitigations:* temperature, top-k, repetition penalty on the generated reply, hard stop at `<|endoftext|>`, and blocking unused padding rows of the embedding table.

## Limitations

* **It is small.** 37M parameters and ~0.5B training tokens is roughly a thousand times less than modern chat models. Answers are often confidently wrong. Do not rely on it for facts.
* **Short memory.** The context is 512 tokens; older turns are dropped when a conversation gets long.
* **Narrow conversation skills.** Fine-tuning data was limited and mostly short exchanges. Long or multi-step conversations drift.
* **English only,** and mostly the kind of text found on web pages linked from Reddit.
* **No safety tuning.** It was trained on unfiltered web text and may produce biased or offensive output.
* **No benchmarks.** I only tracked loss and read samples.

## Reproducing it

1. **Tokenizer:** `BPETokenizer().train_from_files([...], vocab_size=32000)`, then `add_special_tokens(...)` and `save(...)` (`tokenizer/bpe.py`).
2. **Data:** stream OpenWebText to a text file, then `tokenizer.encode_file(in_path, out_path)` (writes a uint16 token file).
3. **Pretrain:** `python train.py` (resumes from `checkpoints/latest.pt` if present).
4. **Fine-tune:** `python finetune.py` (writes `checkpoints/sft_final.pt` and the reasoning tokenizer).
5. **Chat:** `python chat.py`.

## File Structure

```text
tokenizer/bpe.py   byte-level BPE (train, encode, decode, save/load)
model.py           the transformer
train.py           pretraining loop (resume, cosine schedule)
finetune.py        multi-turn SFT with loss masking and embedding resize
chat.py            interactive chat
```

## Credits

The architecture follows the Llama papers; the tokenizer’s split pattern follows GPT-4’s. 
Training data: OpenWebText + Stanford Alpaca. 
All pretraining and fine-tuning ran on Google Colab’s free tier.
