# [PASTE THE ENTIRE BPETokenizer CODE WE WROTE EARLIER HERE]
# Make sure to include the imports, GPT4_SPLIT_PATTERN, bytes_to_unicode, etc.

"""
Byte-level BPE tokenizer (GPT-4 / cl100k style regex pre-splitting).

Features
--------
* Incremental trainer (pair -> word index + lazy max-heap): each merge only
  touches words that contain the merged pair, so 16k-50k merges on a large
  sample is practical in pure Python.
* Streaming training from files (never loads the whole corpus in memory).
* Encode cache, special-token policy, strict decode.
* Atomic, validated JSON save/load.
* Streaming file -> uint16/uint32 .bin encoder with optional multiprocessing
  (Windows safe: call it from under `if __name__ == "__main__":`).

Dependencies: regex, numpy   (pip install regex numpy)
"""
from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator

import numpy as np
import regex as reg

# GPT-4 / cl100k split pattern
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _merge_ids(ids: list, pair: tuple, new_id: int) -> list:
    """Replace every non-overlapping occurrence of `pair` (left to right)."""
    a, b = pair
    out = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def iter_text_blocks(
    path: str, block_chars: int = 1 << 24, encoding: str = "utf-8"
) -> Iterator[str]:
    """Yield the file in large blocks, always cut right after a newline
    (or a space if a block has no newline) so words are never split."""
    with open(path, "r", encoding=encoding, errors="replace") as f:
        buf = ""
        while True:
            data = f.read(block_chars)
            if not data:
                if buf:
                    yield buf
                return
            buf += data
            cut = buf.rfind("\n")
            if cut == -1:
                cut = buf.rfind(" ")
            if cut == -1:
                continue
            yield buf[: cut + 1]
            buf = buf[cut + 1 :]


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
class BPETokenizer:
    def __init__(self, pattern: str | None = None, cache_limit: int = 1_000_000):
        self.pattern_str = pattern or GPT4_SPLIT_PATTERN
        self.pattern = reg.compile(self.pattern_str)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        self._cache: dict[str, list[int]] = {}
        self._cache_limit = cache_limit

    # ------------------------------------------------------------------ info
    @property
    def n_vocab(self) -> int:
        """Embedding table size you need (max id + 1, specials included)."""
        top = max(self.vocab)
        if self.inverse_special_tokens:
            top = max(top, max(self.inverse_special_tokens))
        return top + 1

    # -------------------------------------------------------------- training
    def train(
        self,
        texts: str | Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        """Train from a string or an iterable of strings (resets the tokenizer;
        register special tokens afterwards)."""
        if isinstance(texts, str):
            texts = [texts]
        counts: Counter = Counter()
        for t in texts:
            counts.update(self.pattern.findall(t))
        self._train_from_counts(counts, vocab_size, min_frequency, verbose)

    def train_from_files(
        self,
        paths: str | Iterable[str],
        vocab_size: int,
        max_chars: int | None = None,
        block_chars: int = 1 << 24,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> None:
        """Stream files and train on (at most) `max_chars` characters in total.
        Merge statistics converge fast: 50-200M chars is plenty."""
        if isinstance(paths, str):
            paths = [paths]
        counts: Counter = Counter()
        seen = 0
        for p in paths:
            for block in iter_text_blocks(p, block_chars):
                if max_chars is not None and seen + len(block) > max_chars:
                    block = block[: max_chars - seen]
                counts.update(self.pattern.findall(block))
                seen += len(block)
                if verbose:
                    print(f"  counted {seen:,} chars, {len(counts):,} unique words")
                if max_chars is not None and seen >= max_chars:
                    break
            if max_chars is not None and seen >= max_chars:
                break
        self._train_from_counts(counts, vocab_size, min_frequency, verbose)

    def _train_from_counts(
        self, word_counts: Counter, vocab_size: int, min_frequency: int, verbose: bool
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens, self.inverse_special_tokens = {}, {}
        self._cache.clear()

        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, f in word_counts.items():
            words.append(list(chunk.encode("utf-8")))
            freqs.append(f)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set] = defaultdict(set)
        for wi, w in enumerate(words):
            f = freqs[wi]
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)

        # max-heap with lazy invalidation; ties broken by smallest pair
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        num_merges = vocab_size - 256
        for step in range(num_merges):
            best, best_count = None, 0
            while heap:
                negc, pair = heapq.heappop(heap)
                cur = pair_counts.get(pair, 0)
                if cur <= 0:
                    continue
                if -negc != cur:  # stale entry -> re-insert with fresh count
                    heapq.heappush(heap, (-cur, pair))
                    continue
                best, best_count = pair, cur
                break
            if best is None or best_count < min_frequency:
                break

            new_id = 256 + step
            changed = set()
            for wi in list(where[best]):
                w = words[wi]
                new_w = _merge_ids(w, best, new_id)
                if len(new_w) == len(w):  # stale index entry
                    continue
                f = freqs[wi]
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                for pair in zip(new_w, new_w[1:]):
                    pair_counts[pair] += f
                    where[pair].add(wi)
                    changed.add(pair)
                words[wi] = new_w

            pair_counts.pop(best, None)
            where.pop(best, None)
            changed.discard(best)
            for pair in changed:
                c = pair_counts.get(pair, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, pair))

            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{num_merges}  {best} -> {new_id}  (count {best_count})")

        if verbose:
            print(f"Done. vocab size = {len(self.vocab)}")

    # ------------------------------------------------------- special tokens
    def add_special_tokens(self, tokens: Iterable[str] | dict[str, int]) -> dict[str, int]:
        """Register special tokens. A list gets the next free ids automatically;
        a dict lets you choose ids (must not collide with any existing id)."""
        if isinstance(tokens, dict):
            items = list(tokens.items())
        else:
            nxt = self.n_vocab
            items = []
            for t in tokens:
                if t in self.special_tokens:
                    continue
                items.append((t, nxt))
                nxt += 1
        for tok, tid in items:
            if tok in self.special_tokens and self.special_tokens[tok] != tid:
                raise ValueError(f"Special token {tok!r} already has another id")
            if tid in self.vocab or (
                tid in self.inverse_special_tokens and self.inverse_special_tokens[tid] != tok
            ):
                raise ValueError(f"Special token id {tid} collides with an existing id")
            self.special_tokens[tok] = tid
            self.inverse_special_tokens[tid] = tok
        return dict(self.special_tokens)

    # -------------------------------------------------------------- encoding
    def _bpe(self, ids: list[int]) -> list[int]:
        merges = self.merges
        inf = float("inf")
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, inf))
            if pair not in merges:
                break
            ids = _merge_ids(ids, pair, merges[pair])
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        cache = self._cache
        for chunk in self.pattern.findall(text):
            ids = cache.get(chunk)
            if ids is None:
                ids = self._bpe(list(chunk.encode("utf-8")))
                if len(cache) < self._cache_limit:
                    cache[chunk] = ids
            out.extend(ids)
        return out

    def encode(
        self,
        text: str,
        allowed_special: str | Iterable[str] = "none",
        disallowed_special: str = "text",
    ) -> list[int]:
        """
        allowed_special    : "all", "none", or a set of special-token strings that
                             are converted to their ids.
        disallowed_special : what to do when a *registered but not allowed*
                             special string appears in the text:
                             "text"  -> encode it as ordinary text (default, safe for corpora)
                             "raise" -> raise ValueError
        """
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif allowed_special == "none":
            allowed = set()
        else:
            allowed = set(allowed_special)
            unknown = allowed - set(self.special_tokens)
            if unknown:
                raise ValueError(f"Unknown special tokens: {sorted(unknown)}")

        if disallowed_special == "raise":
            for tok in self.special_tokens:
                if tok not in allowed and tok in text:
                    raise ValueError(f"Disallowed special token {tok!r} found in text")
        elif disallowed_special != "text":
            raise ValueError("disallowed_special must be 'text' or 'raise'")

        if not allowed:
            return self._encode_ordinary(text)

        pat = "(" + "|".join(reg.escape(t) for t in sorted(allowed, key=len, reverse=True)) + ")"
        ids: list[int] = []
        for part in reg.split(pat, text):
            if not part:
                continue
            if part in allowed:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    # -------------------------------------------------------------- decoding
    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[i].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token id: {i}")
        return b"".join(parts).decode("utf-8", errors=errors)

    # ------------------------------------------------------------ save/load
    def _state(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "pattern": self.pattern_str,
            "merges": [[a, b, i] for (a, b), i in sorted(self.merges.items(), key=lambda kv: kv[1])],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def _from_state(cls, data: dict) -> "BPETokenizer":
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported tokenizer file version: {data.get('version')}")
        tok = cls(pattern=data["pattern"])
        expected = 256
        for a, b, idx in data["merges"]:
            if idx != expected:
                raise ValueError(f"Corrupt merges: expected id {expected}, got {idx}")
            if a not in tok.vocab or b not in tok.vocab:
                raise ValueError(f"Corrupt merges: unknown component in ({a}, {b})")
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
            expected += 1
        tok.add_special_tokens(dict(data.get("special_tokens", {})))
        return tok

    def save(self, path: str) -> str:
        if not path.endswith(".json"):
            path += ".json"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state(), f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic
        return path

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        if not path.endswith(".json") and not os.path.exists(path):
            path += ".json"
        with open(path, "r", encoding="utf-8") as f:
            return cls._from_state(json.load(f))

    # -------------------------------------------------- big-corpus encoding
    def encode_file(
        self,
        in_path: str,
        out_path: str,
        num_workers: int = 1,
        block_chars: int = 1 << 22,
        allowed_special: str | Iterable[str] = "none",
    ) -> int:
        """Stream `in_path`, write token ids to `out_path` as a raw .bin
        (uint16 if the vocab fits, else uint32). Returns the token count.
        Read it back with np.memmap(out_path, dtype=..., mode="r").
        With num_workers > 1 call this under `if __name__ == "__main__":` (Windows)."""
        dtype = np.uint16 if self.n_vocab <= 65536 else np.uint32
        allowed = allowed_special if isinstance(allowed_special, str) else sorted(allowed_special)
        total = 0
        with open(out_path, "wb") as out:

            def write(arr: np.ndarray) -> None:
                nonlocal total
                arr.tofile(out)
                total += arr.size

            blocks = iter_text_blocks(in_path, block_chars)
            if num_workers <= 1:
                for b in blocks:
                    write(np.asarray(self.encode(b, allowed_special=allowed), dtype=dtype))
            else:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_worker,
                    initargs=(self._state(), allowed),
                ) as ex:
                    pending: deque = deque()
                    for b in blocks:
                        pending.append(ex.submit(_worker_encode, b))
                        if len(pending) >= num_workers * 2:
                            write(pending.popleft().result())
                    while pending:
                        write(pending.popleft().result())
        return total


# --------------------------------------------------------------------------- #
# multiprocessing workers (module level so they pickle on Windows)
# --------------------------------------------------------------------------- #
_W_TOK: BPETokenizer | None = None
_W_ALLOWED = "none"


def _init_worker(state: dict, allowed) -> None:
    global _W_TOK, _W_ALLOWED
    _W_TOK = BPETokenizer._from_state(state)
    _W_ALLOWED = allowed


def _worker_encode(text: str) -> np.ndarray:
    dtype = np.uint16 if _W_TOK.n_vocab <= 65536 else np.uint32
    return np.asarray(_W_TOK.encode(text, allowed_special=_W_ALLOWED), dtype=dtype)
