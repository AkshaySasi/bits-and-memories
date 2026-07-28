"""Capability control: perplexity on real held-out text per quantization level.

Uses WikiText-2 (raw) as the held-out corpus: real text, not in the Pile
training distribution's memorized set, standard in quantization papers.
Perplexity is computed with a sliding window (stride = half window) over
the concatenated test split, the standard HF evaluation protocol.

Usage:
  python -m quantmem.perplexity --model EleutherAI/pythia-410m-deduped --quant nf4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from quantmem.extract import load_model


@torch.no_grad()
def corpus_perplexity(model, tokenizer, corpus: str = "wikitext",
                      window: int = 1024, stride: int = 512,
                      max_windows: int = 200) -> dict:
    if corpus == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    elif corpus == "pile":
        # real Pile validation-distribution text (first 300 documents)
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        text = "\n\n".join(ds[i]["text"] for i in range(300))
    else:
        raise ValueError(f"unknown corpus {corpus}")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    device = next(model.parameters()).device

    nlls, n_tokens = [], 0
    t0 = time.time()
    starts = range(0, max(1, len(ids) - window), stride)
    for k, s in enumerate(starts):
        if k >= max_windows:
            break
        chunk = ids[s:s + window].to(device)
        target = chunk.clone()
        # only score the second half of each window (avoid double counting)
        scored_from = 0 if s == 0 else window - stride
        target[:scored_from] = -100
        out = model(chunk.unsqueeze(0), labels=target.unsqueeze(0))
        n = int((target != -100).sum()) - 1  # shifted labels
        nlls.append(out.loss.float() * n)
        n_tokens += n
    ppl = float(torch.exp(torch.stack(nlls).sum() / n_tokens))
    return {"perplexity": ppl, "tokens_scored": n_tokens,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    ap.add_argument("--quant", default="fp16")
    ap.add_argument("--corpus", default="wikitext", choices=["wikitext", "pile"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.quant)
    res = corpus_perplexity(model, tokenizer, corpus=args.corpus)
    res.update(model=args.model, quant=args.quant, corpus=args.corpus)
    print(json.dumps(res, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.corpus == "wikitext" else f"_{args.corpus}"
    tag = f"{args.model.split('/')[-1]}_{args.quant}{suffix}"
    (args.out / f"ppl_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"saved results/ppl_{tag}.json")


if __name__ == "__main__":
    main()
