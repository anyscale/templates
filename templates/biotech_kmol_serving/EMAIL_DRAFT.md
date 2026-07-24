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
it. Using your three test compounds, a single L4 does **~4,300 molecules/sec** end-to-end
once the CPU side is scaled out alongside it.

The part I didn't expect: it's bound by **RDKit featurization on CPU, not the GPU**. Two
things convinced me. Doubling the GPUs moved throughput by 2% (4,343 → 4,427/sec). And
running everything on the GPU node instead — so featurization is limited to that node's
cores — drops it to 819/sec, a 5.3× difference from nothing but where the CPU work lands.

That may explain the scaling you were seeing: on a 64-vCPU box, running either 8 CPUs or
the single GPU leaves most of the machine idle, and it's the CPU side that sets throughput.

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

- All three numbers are measured on the same three compounds in `takeda-kmol`, so they are
  directly comparable: **4,343** mol/s (`port/three_twostage_1gpu.json`, 48 CPU featurizer
  replicas + 1 L4), **4,427** on 2 L4 (`port/three_twostage_2gpu.json`), **819** for the
  single-deployment version confined to the GPU node (`port/three_monolith_bulk.json`).
- Deliberately not in the email: the 15,751-molecule library figure (2,809 mol/s). That
  molecule set is **not shipped** — it was built in a workspace we no longer have access
  to — so they cannot reproduce it. The bundled tox21 set is what the quickstart uses.
- The 4,343 figure needs ~12 CPU worker nodes. Confirm `takeda-kmol`'s cap is still raised
  before they log in; at the original cap of 4 the run *hangs* rather than erroring, which
  is the most likely thing to make them think it's broken.
- Confirm invites actually went out and that they can see the workspace.
