"""Join extraction and perplexity results into the memorization-capability table.

Selectivity is the paper's key quantity. For each quantization level q,
relative to the FP32 reference:

  mem_retained(q)  = exact_match(q) / exact_match(fp32)
  cap_retained(q)  = ppl(fp32) / ppl(q)          (<= 1, higher is better)
  selectivity(q)   = log(mem_retained) / log(cap_retained)

selectivity > 1 means memorization decays faster than capability
(quantization is a selective forgetter); ~1 means proportional decay;
< 1 means capability dies first.

Usage:  python -m quantmem.analyze --model pythia-410m-deduped
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

QUANTS = ["fp32", "fp16", "int8", "nf4", "fp4"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pythia-410m-deduped")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for q in QUANTS:
        ef = args.results / f"extract_{args.model}_{q}_s{args.seed}.json"
        pf = args.results / f"ppl_{args.model}_{q}.json"
        if not (ef.exists() and pf.exists()):
            print(f"skipping {q}: missing {'extract' if not ef.exists() else 'ppl'}")
            continue
        e = json.loads(ef.read_text())
        p = json.loads(pf.read_text())
        rows.append({"quant": q, "exact": e["exact_match"],
                     "token_acc": e["token_acc"], "ppl": p["perplexity"]})

    ref = rows[0]
    print(f"\n{args.model}  (reference: {ref['quant']})")
    print(f"{'quant':6} {'extract':>8} {'tok_acc':>8} {'ppl':>8} "
          f"{'mem_ret':>8} {'cap_ret':>8} {'select':>8}")
    out = []
    for r in rows:
        mem = r["exact"] / ref["exact"]
        cap = ref["ppl"] / r["ppl"]
        if r["quant"] == ref["quant"]:
            sel = float("nan")
        else:
            sel = (math.log(max(mem, 1e-9)) / math.log(cap)) if cap < 1 else float("inf")
        out.append({**r, "mem_retained": mem, "cap_retained": cap, "selectivity": sel})
        print(f"{r['quant']:6} {r['exact']:8.3f} {r['token_acc']:8.3f} {r['ppl']:8.2f} "
              f"{mem:8.3f} {cap:8.3f} {sel:8.2f}")

    (args.results / f"summary_{args.model}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.results / f'summary_{args.model}.json'}")


if __name__ == "__main__":
    main()
