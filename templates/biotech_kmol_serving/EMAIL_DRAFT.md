# Draft email to Takeda

**Internal.** Template root, outside the shareable `port/` bundle.

Goal: they log into the workspace and run it themselves. Everything else is context.
Deliberately *not* in the email: latency numbers (23–26 ms per single molecule needs the
batch-size-1 explanation or it reads as a regression against their 5 ms), the ~3,700 mol/s
projection for their box, and any multiple of the undocumented ~170 mol/s baseline.

---

**Subject:** kMoL on Ray Serve — have a go yourself

Hi <name>,

After our call I put the kMoL 5-model molecule ensemble on Ray Serve on a GPU and measured
it. On a single L4 it does **~820 molecules/sec** end-to-end using your three test
compounds, and **~2,800/sec** against a 15,700-molecule size-diverse library once the CPU
side is scaled out.

The part I didn't expect: it's bound by **RDKit featurization on CPU, not the GPU**.
Doubling the GPUs moved throughput by 3%. That may explain the scaling you were seeing — on
a 64-vCPU box, running either 8 CPUs or the single GPU leaves most of the machine idle.

One thing worth flagging: kMoL's Python 3.9 / torch 1.13 stack won't run on a current
datacentre GPU at all, so I ported the molecule inference path to 3.11 and current PyTorch
to match Anyscale's Ray version. It reuses kMoL's architecture and checkpoint format and
matches the original bit-for-bit on CPU.

I've sent you Anyscale invites. Once you're in, start the workspace called **takeda-kmol**
and everything's there:

- **`port/README.md`** — three commands to run it yourself, nothing to install
- **`port/TAKEDA_BRIEF.md`** — the results so far, with the caveats

Idle nodes scale to zero, so it costs nothing sitting there.

Two caveats. The weights are synthetic — right architecture and format, random values — so
throughput is real but predictions are placeholders. Drop your trained `model_0.pt` …
`model_4.pt` into `port/checkpoints/` and it picks them up with no code changes. And
throughput depends heavily on molecule size (~0.285 ms per heavy atom), so if you can send
a heavy-atom histogram of your library I can give you numbers for your actual deck rather
than mine.

Happy to get on a call once you've had a look.

Best,
<you>

---

## Before sending

- Re-verify: ~820 mol/s (`port/three_monolith_bulk.json`, monolith on the three compounds,
  one L4); ~2,800 mol/s (`port/serve_pipeline_results.json`, 48 CPU featurizer replicas +
  1 L4 on the 15,751-molecule library); the +3% GPU-doubling result
  (`port/serve_pipeline_2gpu.json`).
- The ~2,800 figure needs **8 CPU worker nodes**; `takeda-kmol` is currently capped at 4,
  so they cannot reproduce that number as the workspace stands. Either raise the cap before
  they log in, or expect them to reproduce ~820–1,600 instead. The README says so, but it's
  the most likely thing to make them think something is broken.
- Confirm invites actually went out and that they can see the workspace.
