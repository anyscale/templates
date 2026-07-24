# Draft email to Takeda — findings + the two asks

**Internal.** Lives at the template root, outside `port/`, so it is not part of the
shareable bundle. Edit freely; the numbers are all traceable to `port/*_results.json`.

The two asks are (1) their trained `model_0..4.pt`, (2) their library's heavy-atom
histogram. Everything else is context that justifies why those two things unblock us.

---

**Subject:** kMoL on Anyscale — early results, and two things we need from you

Hi <name>,

Following up on last week's call with a first set of results. Short version: we reproduced
the kMoL 5-model molecule ensemble on Anyscale and the scaling problem you described looks
like a CPU-versus-GPU allocation issue rather than anything wrong with Ray Serve. Full
writeup with every number linked to the script that produced it is attached
(`TAKEDA_BRIEF.md`); the highlights:

**The workload is bound by CPU featurization, not by the GPU.** We doubled the GPU capacity
and throughput moved 3% (2,809 → 2,892 molecules/sec). The GNN forward itself runs at
roughly 24,000 molecules/sec on one L4 — about 8× more than the CPU tier can feed it. RDKit
featurization is the constraint, and it is embarrassingly parallel across cores.

**That reframes the "is the GPU worth it" question.** You measured 12 ms/molecule on 8 CPUs
and 5 ms/molecule on the 1 GPU. Both of those were configurations on the same ~64 vCPU box,
which means roughly 56 vCPU were not doing featurization work in either case. You were not
really choosing between 8 CPUs and a GPU; you were using about an eighth of the machine
either way. At the per-core rate we measured, that box projects to ~3,700 molecules/sec.
That is a projection from our hardware, not a measurement on yours — which is part of why
we are asking for the items below.

**The cold starts are checkpoint reloading.** The offline `kmol predict` path loads all five
checkpoints per call. Loading once into long-lived replicas, with a warm-up pass before the
replica is marked healthy, removes it.

**Measured, end to end:** 2,809 molecules/sec on one L4 plus six 8-vCPU CPU nodes, against a
deliberately hard 15,751-molecule set (tox21 + ZINC + ChEMBL above 800 MW). That is
0.36 ms/molecule and roughly $0.32 per million molecules at on-demand list pricing.

Two caveats I want to be upfront about. All of the above runs on **synthetic weights** —
correct architecture and checkpoint format, random values — so throughput and latency are
real but the predictions carry no information. And throughput depends strongly on molecule
size: featurization costs about 0.285 ms per heavy atom, so our numbers move by 2–3× purely
with the size mix of the molecule set.

Which leads to the two things that would help most:

**1. Your trained checkpoints.** If you can share `model_0.pt` … `model_4.pt`, they drop
into `port/checkpoints/` with no code changes — we preserved kMoL's checkpoint format and
class structure specifically so your `state_dict`s load unchanged. That lets us confirm
numerical parity on real weights and give you throughput on a model you recognise. If
sharing weights is awkward, the same check runs on your side with one command and we can
just compare outputs.

**2. Your library's molecule-size distribution.** A heavy-atom-count histogram is enough.
We measured per-size-bucket rates (510 molecules/sec/core under 15 heavy atoms, down to 53
above 60), so your distribution converts directly into expected throughput and cost for
your actual deck instead of ours.

Worth noting on the port itself: kMoL's frozen Python 3.9 / torch 1.13 / CUDA 11.7 stack
cannot run on an L4 at all, so we ported the molecule inference path to current PyTorch. It
reuses kMoL's exact architecture and checkpoint format and matches the original bit-for-bit
on CPU (max abs difference 0.0) and to 1.1e-05 on GPU. That is code equivalence, not an
accuracy result.

Separately, if it is useful before your weights are available, we can train the ensemble on
public tox21 using kMoL's own shipped recipe and dataset so the demo returns plausible
predictions with a reportable ROC-AUC. That would be a real model on public data, not a
model of your endpoints — happy to do it or skip it.

Next on our side is the linear-scaling measurement you said would matter most: a per-node
scaling curve with an external load generator on a realistic request mix, so we can put a
defensible "N nodes gives N× throughput" number in writing rather than inferring it.

Best,
<you>

---

## Notes for whoever sends this

- **Numbers to re-check before sending**, in case later runs change them: 2,809 and 2,892
  mol/s (`serve_pipeline_results.json`, `serve_pipeline_2gpu.json`), ~24,000 mol/s forward
  ceiling (`three_gpu_ceiling.json` — 23,738 on the three test molecules; the older 60,101
  figure was measured on a 10-molecule set and is ~2.5× optimistic), $0.32/M at on-demand
  list.
- **Deliberately not claimed:** any multiple of the "~170 mol/s baseline" (undocumented,
  see `cleanup.md`), near-linear multi-node scaling on the real molecule set (not yet
  measured), and anything about accuracy.
- The ~3,700 mol/s projection for their box is the most persuasive line in the email and
  the least substantiated. It is labelled as a projection here; keep it that way.
- Their three routine test molecules (minoxidil, sildenafil, atorvastatin) are already
  measured in the brief, which is a good detail to mention verbally — it shows we listened.
