"""Measure verbatim memorization extraction under quantization.

Protocol (matches the EleutherAI Pythia memorization evaluation):
  * each record is a 64-token sequence from the Pile that the FP16 model
    reproduces greedily: 32-token prompt -> 32-token continuation
  * we prompt the (quantized) model with the first 32 tokens, greedily
    decode 32 tokens, and compare to the true continuation

Metrics per configuration:
  * exact_match: fraction reproduced perfectly (the extraction rate)
  * mean_prefix: mean number of leading tokens correct (partial memory)
  * token_acc:  mean fraction of the 32 tokens correct

Usage:
  python -m quantmem.extract --model EleutherAI/pythia-410m-deduped \
      --quant fp16 --n 2000 --seed 0
  --quant one of: fp16, int8, nf4, fp4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

PROMPT_LEN = 32
CONT_LEN = 32

MEM_FILES = {
    "EleutherAI/pythia-160m-deduped": "data/deduped.160m-00000-of-00001-9973cda220809f72.parquet",
    "EleutherAI/pythia-410m-deduped": "data/deduped.410m-00000-of-00001-4bb233eb8d735ee8.parquet",
    "EleutherAI/pythia-1b-deduped": "data/deduped.1b-00000-of-00002-547b2b00c0f47827.parquet",
}


def load_model(name: str, quant: str):
    kwargs = {"low_cpu_mem_usage": True}
    if quant == "fp16":
        kwargs.update(dtype=torch.float16, device_map="cuda")
    elif quant == "fp32":
        kwargs.update(dtype=torch.float32, device_map="cuda")
    elif quant == "int8":
        from transformers import BitsAndBytesConfig
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                      device_map="cuda")
    elif quant in ("nf4", "fp4"):
        from transformers import BitsAndBytesConfig
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant,
            bnb_4bit_compute_dtype=torch.float16,
        ), device_map="cuda")
    elif quant in ("rtn4", "rtn8"):
        # independent, dependency-free control: group-wise round-to-nearest
        # applied to the weights, orthogonal to bitsandbytes' NF4/FP4.
        kwargs.update(dtype=torch.float16, device_map="cuda")
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        rtn_quantize_(model, bits=4 if quant == "rtn4" else 8, group_size=64)
        model.eval()
        return model
    else:
        raise ValueError(f"unknown quant {quant}")
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model


@torch.no_grad()
def rtn_quantize_(model, bits: int, group_size: int = 64) -> None:
    """In-place group-wise symmetric round-to-nearest of all Linear weights.

    Each row is split into groups of `group_size`; every group is scaled by
    its max absolute value, rounded to `bits`-bit signed integers, and
    dequantized back to fp16. This reproduces the exact quantization error
    a naive weight-only quantizer would incur, at full inference speed.
    """
    qmax = 2 ** (bits - 1) - 1
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            out_f, in_f = w.shape
            pad = (-in_f) % group_size
            if pad:
                w = torch.nn.functional.pad(w, (0, pad))
            g = w.reshape(out_f, -1, group_size)
            scale = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / qmax
            gq = torch.round(g / scale).clamp(-qmax - 1, qmax) * scale
            wq = gq.reshape(out_f, -1)[:, :in_f]
            module.weight.data.copy_(wq.to(module.weight.dtype))


def load_memorized(model_name: str, n: int, seed: int) -> np.ndarray:
    """Sample n memorized sequences, caching the small result to avoid holding
    the full 800k-row parquet in memory on later (large-model) runs."""
    tag = model_name.split("/")[-1]
    cache = Path("data_raw") / f"sampled_{tag}_n{n}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)

    from huggingface_hub import hf_hub_download
    path = hf_hub_download("EleutherAI/pythia-memorized-evals", MEM_FILES[model_name],
                           repo_type="dataset", local_dir="data_raw")
    df = pd.read_parquet(path, columns=["tokens"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    seqs = np.stack(df.iloc[idx]["tokens"].to_numpy()).astype(np.int64)
    del df
    assert seqs.shape[1] == PROMPT_LEN + CONT_LEN
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, seqs)
    return seqs


@torch.no_grad()
def extraction_metrics(model, seqs: np.ndarray, batch_size: int = 32) -> dict:
    device = next(model.parameters()).device
    exact, prefix_lens, token_accs = 0, [], []
    t0 = time.time()
    for i in range(0, len(seqs), batch_size):
        batch = torch.from_numpy(seqs[i:i + batch_size]).to(device)
        prompts, truth = batch[:, :PROMPT_LEN], batch[:, PROMPT_LEN:]
        out = model.generate(
            prompts,
            attention_mask=torch.ones_like(prompts),
            max_new_tokens=CONT_LEN,
            do_sample=False,
            num_beams=1,
            pad_token_id=0,
        )
        gen = out[:, PROMPT_LEN:PROMPT_LEN + CONT_LEN]
        match = (gen == truth)
        exact += int(match.all(dim=1).sum())
        # longest correct prefix per row
        first_wrong = (~match).float().argmax(dim=1)
        all_right = match.all(dim=1)
        plen = torch.where(all_right, torch.full_like(first_wrong, CONT_LEN),
                           first_wrong)
        prefix_lens.extend(plen.cpu().tolist())
        token_accs.extend(match.float().mean(dim=1).cpu().tolist())
    n = len(seqs)
    return {
        "n": n,
        "exact_match": exact / n,
        "mean_prefix": float(np.mean(prefix_lens)),
        "token_acc": float(np.mean(token_accs)),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    ap.add_argument("--quant", default="fp16")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    seqs = load_memorized(args.model, args.n, args.seed)
    print(f"loaded {len(seqs)} memorized sequences for {args.model}")
    model = load_model(args.model, args.quant)
    res = extraction_metrics(model, seqs, args.batch)
    res.update(model=args.model, quant=args.quant, seed=args.seed)
    print(json.dumps(res, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}_{args.quant}_s{args.seed}"
    (args.out / f"extract_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"saved results/extract_{tag}.json")


if __name__ == "__main__":
    main()
