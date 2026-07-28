# Bits and Memories

Code and data for the paper *Bits and Memories: Measuring Verbatim Extraction
Across LLM Quantization*.

Deployed language models are almost always quantized, and people increasingly
ask whether quantization also lowers privacy risk. Prior work answers that
question with membership inference. We think the risk that actually matters is
a model reproducing its training data word for word, so we measure that
directly: verbatim extraction of known-memorized sequences across five
precision levels and three model sizes, with a perplexity control at every
point.

Two findings:

1. **Quantization forgets memorization faster than capability.** At every
   size and precision, verbatim extraction drops faster than perplexity
   rises. The effect holds under two unrelated quantizers (bitsandbytes NF4
   and our own round-to-nearest) and two corpora (WikiText-2 and the Pile).
2. **But it is not a privacy defense.** At 1B parameters, 4-bit quantization
   keeps ~95% of capability while still reproducing ~72% of memorized
   sequences, and the surviving fraction *grows* with model size.

Everything here runs on a single consumer GPU (developed on a GTX 1650 Ti,
4 GB).

## What is real here

Nothing is synthetic. We use the Pythia models, whose training corpus (the
Pile) is public, and the officially published set of sequences each model is
known to have memorized. We do not guess what a model memorized; we use the
released ground-truth list.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install torch transformers datasets bitsandbytes numpy pandas matplotlib
```

CUDA is used when available; the runs also work on CPU, more slowly.

## Reproducing the paper

```bash
# extraction: 32-token prompt -> 32-token greedy continuation, exact match
python -m quantmem.extract --model EleutherAI/pythia-410m-deduped --quant nf4 --n 500

# capability control: perplexity on held-out text
python -m quantmem.perplexity --model EleutherAI/pythia-410m-deduped --quant nf4 --corpus pile

# --quant is one of: fp32 fp16 int8 nf4 fp4 (bitsandbytes) | rtn8 rtn4 (our independent method)
# --corpus is one of: wikitext pile

# join extraction + perplexity into the selectivity table
python -m quantmem.analyze --model pythia-410m-deduped

# figures
python -m quantmem.figures       --out figures
python -m quantmem.method_figure --out figures/method.png
```

The first extraction run for a model downloads the memorized-sequence set and
caches a fixed 500-sequence sample, so every precision level is scored on
identical data.

## Layout

```
quantmem/
  extract.py         quantization (incl. our RTN quantizer) + verbatim extraction
  perplexity.py      sliding-window perplexity on WikiText-2 and the Pile
  analyze.py         extraction + perplexity -> selectivity table
  figures.py         extraction curve and capability-memorization plane
  method_figure.py   the protocol diagram
results/             per-configuration JSON results used in the paper
sampled_data/        the exact 500-sequence samples evaluated (one per model)
figures/             generated figures
paper/main.tex       the paper source
```

## The RTN quantizer

To show the effect is not an artifact of one library, `extract.py` includes a
short, dependency-free group-wise round-to-nearest quantizer (`rtn4`, `rtn8`)
applied directly to the linear weights. It shares no code with bitsandbytes,
and it reproduces the same selective forgetting.

## Citation

```bibtex
@misc{sasi2026bitsandmemories,
  title  = {Bits and Memories: Measuring Verbatim Extraction Across LLM
            Quantization},
  author = {Sasi, Akshay},
  year   = {2026},
}
```

## License

MIT. See LICENSE.
