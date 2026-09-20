# recreator

**Rebuild a pretrained transformer as a [HELIX](https://pypi.org/project/helix-lm/) model without pretraining it again.**

```bash
pip install recreator
```

```python
from recreator import recreate, freeze_to_new_strands

result = recreate("Qwen/Qwen3-8B", preset="balanced")
freeze_to_new_strands(result.model)
```

---

## The idea

A transformer block and a HELIX block are mostly the same block. Both carry
`q_proj`/`k_proj`/`v_proj`/`o_proj`, the per-head `q_norm`/`k_norm` Qwen3 introduced, a SwiGLU
`gate_proj`/`up_proj`/`down_proj`, two RMS norms, an embedding table and a head. HELIX braids two
extra sequence mixers into the same residual stream — a gated delta-rule recurrence (**strand R**)
and a hierarchical landmark index (**strand I**) — and *those* are the only parts a donor cannot
supply.

So swapping a model's architecture is not a retraining problem. It is a graft plus a repair:

```
donor transformer
      │  copy every tensor that already means the same thing   (~78% of the student)
      ▼
HELIX student with two silent strands
      │  train only what the donor could not supply            (~16-25% of parameters)
      ▼
HELIX model carrying the donor's knowledge
```

## Why a naive graft fails, and what fixes it

Copying the weights is the easy half. Do only that and the result is *worse than useless*: strands
R and I are still at their random initialisation, writing noise into a residual stream the donor's
weights were tuned against. The measured cost on a 4-layer donor:

| | KL from donor | argmax agreement |
| --- | --- | --- |
| weights copied, strands left random | 0.289 | 34.4% |
| weights copied, strands **silenced** | **1.7e-07** | **100%** |

The fix is to make both new strands contribute *exactly zero* at step zero while keeping a live
gradient path, so training starts at the donor's loss instead of near noise:

- **Strand R** ends in `out_proj`. Zeroing that one matrix makes its contribution identically zero.
  Its input is not zero, so the gradient with respect to `out_proj` is not zero either — the strand
  learns its way back in rather than being cut off.
- **Strand I** is *not* additive. It is mixed with local attention through a two-way softmax, so
  zeroing the gate gives `softmax([0,0]) = [0.5, 0.5]` — a half-and-half blend with an untrained
  index, which is the opposite of silent. HELIX builds that gate without a bias, so `recreator`
  swaps in a bias-carrying `Linear`, zeroes the weight and sets the bias to `[+b, -b]`. The gate
  starts at `sigmoid(2b) ≈ 0.9999` on local attention and is free to open the index as it trains.

The claim this rests on is falsifiable, so the package tests it:

```bash
recreator verify Qwen/Qwen3-8B --preset faithful
```

```
identity graft check at 1024 tokens (local span 1024):
  max abs difference  6.223e-05
  relative error      1.850e-05  (tolerance 1.0e-03)
  KL(donor || graft)  1.629e-06
  argmax agreement    100.00%
  PASS
```

Inside strand L's window the grafted HELIX model **is** the donor, to floating-point noise. Beyond
that window the donor attends globally and strand L does not — that divergence is real, and closing
it is exactly what the training stage is for.

## Does the repair work?

Grafting a donor into a deliberately crippled student — a 32-token local window, so nearly all of
the donor's long-range behaviour is out of reach — and then training **only** the new strands:

| | KL from donor |
| --- | --- |
| after graft | 4.76 |
| after 120 steps | **0.36** (−92%) |

24.9% of parameters trained. The rest never moved.

## What fits on your card

Guessing wastes hours, so `recreator plan` estimates every line before you start:

```bash
recreator plan Qwen/Qwen3-8B --budget 96 --seq-len 4096
```

```
student     9.73B parameters
trainable   1.54B (15.8%)
teacher     8.19B parameters

  student weights      18.13 GiB
  gradients             2.87 GiB
  optimizer state      17.22 GiB
  teacher (frozen)      4.20 GiB     ← NF4, or 0 with --cache-teacher
  activations           1.53 GiB
  TOTAL                43.95 GiB
  budget               96.00 GiB

FITS: 52.05 GiB spare
```

The trainable fraction is why this fits where full fine-tuning does not: gradients, moments and
master weights all scale with *trainable* parameters, and a graft freezes everything the donor
supplied. A plan that does not fit prints the ranked list of what to change instead of just failing.

## MoE donors

A HELIX block has one dense MLP; GPT-OSS and Mixtral have many plus a router. There is no exact
dense equivalent, but there is a right way and a wrong way to approximate one.

Averaging expert weights is the wrong way, and not by a little — `expert(mean(W))` is not
`mean(expert(W))` once a SwiGLU sits between them. **Concatenating** them is exact: stack the kept
experts' `gate`/`up` rows and their `down` columns and the block-diagonal structure makes one dense
layer compute `Σᵢ wᵢ · expertᵢ(x)` — the true routed mixture.

| merge mode | error vs. the real mixture |
| --- | --- |
| `average` (weight blending) | 2.4e+02 — on a signal of magnitude 2.6e+02 |
| `concat` (default) | **6.1e-05** |

The price is width: keeping `k` experts of width `m` gives an MLP of width `k·m`. `recreator`
defaults to the donor's `num_experts_per_tok`, so the dense student keeps the MoE's *active* width.

> **Worth knowing before you plan a GPT-OSS run.** GPT-OSS-120B has ~117B parameters but only ~5B
> active per token, because the experts hold nearly all the mass. Collapsing it to a dense student
> gives a model of a few billion parameters, not 120B. That is not a limitation of this tool — it is
> what "dense equivalent of a sparse model" means. You are distilling a 120B *teacher* into a small
> dense student, which is a legitimate and useful thing to do, but it is not training a 120B model.
> `recreator plan` prints the real number before you commit to it.

## Fitting more than the card holds

- **`--cache-teacher`** — the teacher is frozen, so it need not be resident. Run it once over the
  corpus, store the top-k logits to disk, and replay them. The teacher's memory becomes the
  student's. Top-k is lossy but well-behaved: `sparse_kl_loss` renormalises over the kept entries,
  so the target stays a proper distribution.
- **Layer-wise training** — a HELIX block and a transformer block have the same interface, so block
  *i* can be trained to reproduce block *i* of the teacher in isolation, with no gradient crossing a
  block boundary. Peak memory becomes a function of one block rather than the whole model. It cannot
  fix errors that only appear when blocks compose, so the usual sequence is layer-wise first, then a
  short end-to-end pass.
- **Context ladder** — strand I retrieves across long context, and at short lengths there is nothing
  to retrieve: every token is already inside strand L's window and the index has no closed blocks to
  descend. Training long from the start pays for that context on every step while the student is
  still learning the easy part. The ladder starts short and climbs, block-aligned throughout.

## Full pipeline

```python
import torch
from recreator import (
    recreate, freeze_to_new_strands, distill, DistillConfig,
    ContextLadder, load_teacher, save_recreated, push_to_hub,
)

result = recreate("Qwen/Qwen3-8B", preset="balanced", dtype=torch.bfloat16)
print(result.report.summary())

teacher = load_teacher("Qwen/Qwen3-8B", load_in_4bit=True)
trainable, total = freeze_to_new_strands(result.model)
print(f"training {trainable/total:.1%} of parameters")

record = distill(
    result.model.cuda(), batches, teacher=teacher,
    config=DistillConfig(
        steps=2000, learning_rate=2e-4, eight_bit_optimizer=True,
        ladder=ContextLadder(start=512, end=32768, rungs=6, block_size=64),
    ),
)

save_recreated(result.model, "out/", donor_path=result.donor_id,
               donor_id="Qwen/Qwen3-8B", report=result.report, record=record)
push_to_hub("out/", "you/qwen3-8b-helix", private=True)
```

## CLI

| command | what it does |
| --- | --- |
| `recreator plan <donor> --budget 96` | estimate memory before committing |
| `recreator graft <donor> --verify -o out/` | build the student, check it, save it |
| `recreator verify <donor>` | check a graft reproduces its donor |

Presets: `faithful` (widest window, closest to the donor), `balanced` (default),
`long_context` (short windows, heavy retrieval).

## Install

```bash
pip install recreator            # graft and plan
pip install "recreator[train]"   # + transformers, datasets (distillation)
pip install "recreator[quant]"   # + bitsandbytes (4-bit teacher, 8-bit optimizer)
pip install "recreator[all]"
```

## Honest limits

- A graft is exact **inside the local window** and approximate outside it. Recovering long-range
  behaviour is training, not copying, and how well it recovers depends on your corpus and budget.
- Donor and student share a tokenizer. Grafting across vocabularies copies the shared rows and
  leaves the rest at init — workable when vocabularies overlap heavily, not a general solution.
- MoE → dense loses capacity that no merge recovers, as above.
- The numbers here come from small synthetic donors in the test suite. They verify that the
  *mechanisms* do what they claim; they are not benchmark results for an 8B recreation.
- `plan` is an estimate with documented assumptions, not a measurement.

## Running the tests

```bash
pip install "recreator[dev]" && pytest
```

The suite builds a small Qwen3-shaped donor on disk and checks every claim above — the exact graft,
the ablation that shows why silencing matters, the exact expert merge, and that distillation
actually closes the gap.

## License

Apache-2.0. A recreated model inherits its donor's license; `push_to_hub` defaults to private
because that call belongs to you, not to this tool.
