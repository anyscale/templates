# Scaling tiers — a practical level-setting model

Working notes for the nb 09 "scaling up" narrative and a deck level-setting slide.
The goal is a *practical* definition of "scale" grounded in this workload
(transaction foundation model on Ray/Anyscale), stated generically enough to reuse.

## The core idea

**A tier isn't a size, it's a wall.** Each boundary is defined by the specific thing
that *breaks* and forces you to acquire a capability you didn't need before. That's why
the jumps feel uneven: a 10× data increase *inside* a tier is boring, but crossing a wall
changes how you work. Sizing by "the wall you just hit" is more honest and more useful to
an audience than GB or GPU counts — it tells them *why* they'd move, not just that things
get bigger.

## The tiers

| Tier | Typical hardware | What it's for | The wall that ends it |
|---|---|---|---|
| **Laptop** | CPU or an integrated/tiny GPU | Prove the code is correct; fast iteration; CI | No accelerator, or it won't fit in RAM / is too slow to learn anything real |
| **Single GPU** | One card — the *range* is huge: an old 16 GB T4 up to an 80 GB H100 | Smallest *real* training; does the model actually learn | Model + activations won't fit one card's memory, or one card is too slow to iterate |
| **Single node, multi-GPU** | 2–8 GPUs in one box (DGX, `p4d`/`p5`, `g5.48xlarge`) | Bigger models, faster runs; the cheapest place to *learn* distribution | One box's aggregate memory or wall-clock is the ceiling |
| **Multi-node** | 10s–1000s of GPUs across many boxes | Workloads that exceed a single machine | Cost — the run gets big enough that someone has to approve it |
| **Fleet / production** | Same, but committed and long-running | The workload is a product dependency, not an experiment | (No wall — this is where you optimize to stay affordable and survivable) |

## What the model gets right

**The single-GPU → single-node hop is a small hardware step but a real software barrier.**
You cross from one process to *collective communication* — the training loop has to become
distribution-aware (DDP/FSDP, gradient sync). But it's the cheapest place to learn that,
because you're spared the hard parts: one filesystem, one failure domain, a fast interconnect
(NVLink ~hundreds of GB/s). Nothing can partition. You pay the "go parallel" tax without the
"go distributed" tax.

**The top-tier checklist is forced, not optional** — utilization maxing, checkpointing,
elastic failure recovery, spot/preemptible for cost, observability. The grounding is simple
failure math: if each GPU-hour carries some small independent chance of a fault, the
probability that *something* dies during a run climbs toward 1 as (GPUs × hours) grows. At
thousands of GPUs over weeks, hardware failure is a constant, not an event — published
large-model training logs are full of restarts. Checkpointing and auto-recovery become the
only way the run finishes at all. Spot follows the same logic: once you're already engineering
for random failure, taking the 60–70% discount is nearly free.

## Two refinements

**The multi-node entry is arguably the biggest technical wall, not a smooth continuation.**
Scaling multi-node *bigger* is gated by cost, but *getting to multi-node at all* is where you
first cross the network: slower interconnect (EFA/InfiniBand at 100s of Gb/s, still an order of
magnitude under NVLink), multiple failure domains, and the need for an orchestrator, distributed
storage, and shared coordination. This is the tier where Ray/Anyscale earns its keep — and, to
keep the story honest, where it *starts* to. At laptop / single-GPU / single-node, Ray is
convenient but not differentiated; the differentiation is orchestration, fault tolerance, elastic
scaling, and observability, all of which are multi-node problems. Say that plainly rather than
claiming Ray transforms the laptop tier.

**The "business line" is two boundaries with a regime flip between them:**
technical (walls 1–3) → economic (the approval wall) → technical-again-but-now-justified-by-economics
(the production regime). The one-liner: *scale stops being a "can we?" question and becomes a
"should we spend it?" question — and once the answer is yes, efficiency becomes a technical mandate,
because every idle GPU-hour is now real money.*

The narration arc for the slide:
*Does it run? → does it run on a GPU? → can it use more than one? → can it cross the network? →
can we afford it? → can we run it efficiently and survive failures at that cost?*

## Grounding the business line

It's a range, but honest anchors: on-demand, a single GPU is ~$1–4/hr, an 8-GPU node ~$30–100/hr.
So a single-GPU day is tens of dollars, a single-node day is hundreds to low-thousands, and a dozen
nodes for a week is easily six figures. The clearest signal of the wall isn't a dollar amount, it's a
*procurement behavior*: the first time you have to **reserve capacity ahead of time** (a capacity
block / reservation) instead of grabbing GPUs on demand. Reservation means commitment; commitment
means finance is in the loop. Equivalent heuristics: the first time "what'll this cost?" gets asked
*before* you hit run, or when a single run rivals a headcount-month of salary. Below that it's team
budget and nobody asks; above it, it's a line item.

## Grounding it in this workload

The pretrain run itself is a live demo of the multi-node wall. Because each node here is 1×A10G,
the 8-worker job is *8 separate single-GPU nodes coordinating across the network* — the multi-node
tier, not single-node. What we watched during provisioning — the PACK placement group blocking
`fit()` until all 8 workers are up, a 4×A10G node launch failing, the autoscaler assembling nodes one
at a time — *is* the new coordination-and-failure modes the tier introduces, happening in real time.
Real, un-staged evidence for the slide. (See `PERFORMANCE.md` for the captured events.)
