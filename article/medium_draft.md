# I Asked Whether Shrinking an AI Also Makes It Forget Its Secrets

*Everyone compresses their language models before deploying them. A lot of
people quietly hope this also scrubs the private data the model memorized. I
tested that hope on real models. The answer is: sort of, but not really, and
it gets worse the bigger the model.*

---

## The thing nobody quite says out loud

Two facts about large language models are both true, and they make an
awkward pair.

The first: models memorize. If a phone number, a private email, or a page of
a copyrighted book showed up enough times in the training data, the model can
often be coaxed into typing it back out word for word. This isn't a rumor;
it's a well-studied phenomenon with a decade of papers behind it.

The second: almost nobody runs a model at full size. Before deployment, we
*quantize* it — store its numbers at lower precision, 8 bits or 4 bits
instead of 16 or 32, so it fits on cheaper hardware. Your laptop can run a
model that used to need a server, because someone rounded off its weights.

Now put those two facts in the same room. When you round off a model's
weights, what happens to the stuff it memorized? There's a comfortable
assumption floating around — that a compressed model is a *safer* model,
that squeezing it must also blur out the private data. I wanted to know if
that was actually true, or just something we tell ourselves.

## Why the usual test is the wrong test

Here's where I want to plant a flag, because it's the part I care about most.

Most of the recent work on "does compression help privacy" measures
something called **membership inference**: can an attacker figure out whether
a specific document was *in* the training set? And several papers report,
reasonably, that quantization makes membership inference harder. Good news,
right?

I think that's the wrong question. Knowing a document was *in* the training
data is a world away from being able to *print the document*. One is a hint;
the other is the leak. If a model can still recite someone's private data
verbatim, it doesn't matter that a membership-inference attack got a little
weaker. The recital is the harm.

So I measured the recital directly. Not "was this in training?" but "can the
compressed model still type it back out, exactly?"

## How you test this without guessing

The tricky part of studying memorization is that you usually don't *know*
what a model memorized. You'd be guessing.

Except for one family of models. The **Pythia** models were built for
research: their entire training set is public, and — this is the gift —
researchers already published the exact list of sequences each model is known
to have memorized. No guessing. I could take that official list, show the
model the first half of a sequence it definitely memorized, let it continue,
and check whether the second half comes back *exactly*.

Then I did that at every precision level — full precision, 8-bit, 4-bit — and
at three model sizes. And at every single point I also measured the model's
general ability (perplexity on ordinary text), because otherwise I couldn't
tell "the model forgot this specific passage" apart from "I broke the model."

That control turns out to be the whole ballgame.

## Finding one: compression forgets memories faster than skills

The first result is kind of elegant.

When you quantize hard enough, both things degrade: the model gets a bit worse
at everything, *and* it forgets some of what it memorized. But they don't
degrade at the same rate. **Memorization drops faster than general ability.**
Every time. Across every model size, every precision, and — importantly — with
two completely different quantization methods, including a bare-bones one I
wrote myself from scratch so nobody could say it was a quirk of one library.

I like to picture it as fog rolling in. A model's *skills* are stored
redundantly, spread across many weights, so a little rounding barely touches
them. Its *verbatim memories* seem to live in delicate, precise
configurations — and rounding smudges those first. Compression fogs up the
photographic memory long before it dulls the intelligence.

If the story ended there, it'd be a feel-good result: "compress your models,
lose the memorized secrets faster than the usefulness." Publish, go home.

## Finding two: it still isn't a privacy defense — and scale makes it worse

The story does not end there.

Yes, memorization dies faster than capability. But *faster* is not the same
as *enough*. Watch what happens as the models get bigger.

At the smallest model I tested, aggressive 4-bit compression really did wipe
out most of the memorized data. Encouraging. At the middle size, still
decent. But at the largest model — one billion parameters — 4-bit compression
kept **95% of the model's ability while still coughing up 72% of the memorized
sequences**. Almost none of the usefulness gone, almost all of the memory
still there for the taking.

And the trend across the three sizes points the wrong way: the fraction of
memorized data that *survives* compression went up as the models got bigger.
Larger models are so over-provisioned that a bit of rounding barely disturbs
them, memories included. The models people actually deploy are a hundred times
bigger than my biggest one. Nothing in my data suggests the leak politely
shrinks again up there. If anything, it grows.

So the honest bottom line, the sentence I'd want a deploying engineer to read:
**don't count on quantization to remove memorized training data. It doesn't,
and it removes less of it the bigger your model gets.**

## The part that surprised me

The result I didn't expect was how *cleanly* the two things separated. There's
a real, consistent gap between how fast a model loses its memories and how
fast it loses its skills — a gap you can put a single number on, and one that
widens with scale. To me that's a small clue about something deeper: that
memorization and general ability aren't the same substance smeared across the
weights. They seem to live in physically different places, one fragile and one
robust. There's a recent result showing that quantizing an "unlearned" model
can actually bring the erased knowledge *back*, which fits the same picture:
memories are stubborn, tucked into precise corners of the weights that
rounding disturbs but doesn't reliably destroy.

That's not a privacy paper anymore; that's a hint about how these things
store what they know. I think it's worth chasing.

## The honest caveats

I ran this on one model family (the only one with public memorization ground
truth), topping out at a billion parameters, on English text, with one style
of quantization and the strictest definition of "extraction." So the specific
numbers are scoped, and I say so plainly in the paper. The *direction* —
memories die faster than skills, but not fast enough, and less so at scale —
is consistent enough that I'd bet on it holding up. But bets aren't proofs,
and the obvious next experiment is simply: bigger models. The code is set up
to run exactly that, the moment someone has the GPUs.

## The whole thing ran on a gaming laptop

One more note, because I think it matters. None of this needed a cluster. The
entire study ran on a consumer GPU with 4 GB of memory — the kind of card
people buy to play games. The models are public, the memorized-data list is
public, and the hardest part was not compute; it was asking a slightly
different question than everyone else and measuring it carefully.

There's a lesson in that I keep relearning: a lot of good questions are
sitting in the gap between two well-studied things, waiting for someone to
put them in the same room.

## Try it yourself

Everything is public and reproducible:

- **Paper (arXiv):** [ARXIV_LINK]
- **Code:** https://github.com/AkshaySasi/bits-and-memories
- **Sampled data + results:** https://huggingface.co/datasets/AkshaySasi/bits-and-memories

Pick a Pythia model, quantize it to 4 bits, and watch how much it still
remembers. It's a strange feeling, prompting a shrunk-down model with half of
someone's memorized text and watching the other half come back anyway.

---

*Akshay Sasi is an AI/ML engineer. If you run this at larger scale and the
leak finally shrinks, I'd genuinely love to be wrong — tell me.*
