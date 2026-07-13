# critiques.md — running review of the autoresearch work

Maintained by a second Claude session watching the branch. Casual on purpose — plain
words, simple analogies. Each round is stamped with what it reviewed, so we know
what's been looked at and what hasn't.

---

## Round 1 — 2026-07-11, baseline (nothing committed yet)

**Reviewed:** the 4 untracked files as of today — `README.md`, `REQUIREMENTS.md`,
`BUDGET_POLICY.md`, `campaigns/_TEMPLATE.md`. The README promises `SEED_INDEX.md` and
9 seed plans that don't exist yet, so he's clearly mid-flight. Holding judgment on those.

### The big picture (what I think he's doing, in one breath)

He took the one fraud-model campaign you two ran by hand and is writing the "franchise
manual" so you can run 9 more like it: a rulebook for spending money (BUDGET_POLICY), a
spec for the robot lab assistant that doesn't exist yet (REQUIREMENTS R1–R9), and a
fill-in-the-blanks form for starting each new project (the campaign template).

**Overall: the thinking is genuinely good.** The scientific hygiene — "reproduce their
number before you claim anything," "one experiment = one flag = one commit," "every
number gets error bars" — is the real deal, and it's all traceable to scars from the
actual campaign, not invented rules. My critiques are mostly about *sequencing* and a
few load-bearing details that don't hold up if you poke them.

### 🔴 Critique 1: He's writing the cookbook before cooking the second meal

Everything so far is documents about software that doesn't exist. REQUIREMENTS.md even
says so: "nothing here is built yet."

**ELI5:** You went on one great road trip, and he's now writing a 40-page fleet-operations
manual before you've bought a second car. The trip was great! But you learned to drive
*on the trip*, not from the manual.

The danger with generalizing from n=1 is that you can't tell which rules were
load-bearing and which were just... what happened to work that week. A second campaign
run with a *minimal* harness would tell you which of R1–R9 actually earn their keep.

**Fix I'd suggest:** freeze the docs after the seed plans land, then build only R1 (the
results ledger) + R2 (the launcher) as a couple hundred lines of Python, and run the
cheapest wave-1 campaign with just that. Let campaign #2 rewrite the requirements. His
own build-order section almost says this — it just needs a forcing rule: *no more docs
until `registry.py` exists.*

### 🔴 Critique 2: The budget cap is a pinky promise, not a prepaid card

R4 says the launcher "estimates a run's GPU-hours *before* submit and refuses runs that
exceed the cap." But the docs also admit estimates are **±50%**. So the bouncer checks
your ID at the door... and then nobody watches what you do inside.

**ELI5:** It's a parent saying "promise you'll only spend $10 at the arcade" vs. handing
the kid a prepaid card with $10 on it. Only one of those actually caps spend.

**The missing piece is a runtime kill:** every generated job should carry a hard
wall-clock timeout derived from the cap (est. hours × safety margin → job `timeout`).
Estimate at submit + hard abort at runtime = a real cap. Right now R4 only has the
estimate half, and R3's canaries only catch *broken* runs, not *slow* ones. This is a
one-line addition to R2's generated job spec and it converts the whole budget policy
from "advisory" to "enforced."

### 🟠 Critique 3: R1 says decisions must be machine-readable, then stores them in a prose field

R1's whole pitch: promote/kill decisions "read the registry programmatically, not
prose." But look at the schema — the only place a decision lives is
`"notes": "beats baseline...; promote to full"`. That's prose! The requirement
contradicts its own example.

**Fix:** either a `decision: "promote" | "kill" | "hold"` field on the row, or (cleaner)
decisions get their own row type in the same JSONL — runs are facts, decisions are
judgments, and blending them in a free-text field means the "fresh agent resumes from
disk alone" promise quietly breaks for the most important bit: *why* things were killed.

### 🟠 Critique 4: Two different components both claim to write the registry row

- R1: the row is "written by the run itself to durable storage."
- R6: the monitor, "on *any* terminal state... writes the R1 row."

Both can't own it. And R1's version has a hole: a run that crashes hard (spot preemption,
OOM, node dies) *can't* write its own `status: FAILED` row — dead people don't file their
own death certificates. The monitor has to own terminal-state rows.

**Fix:** pick one writer. Cleanest: the run writes a `RUNNING`/heartbeat row with
metrics as it goes; the monitor writes the single terminal row. Or make writes idempotent
keyed on `run_id` + `status`. Either way, say it — otherwise campaign #2 gets duplicate
or missing rows and the "sum gpu_hours to enforce the envelope" math (R4 depends on it!)
silently double-counts.

### 🟠 Critique 5: The proxy-rung cap and the wave-1 envelope don't leave room to breathe

Proxy cap is ≤10 GPU-hr *per idea*. Wave-1 total envelope is ≤60 GPU-hr. So six proxy
ideas at the cap = the entire campaign, with zero left for the gates, the full runs, or
the long tail of controls the docs themselves say to budget for.

**ELI5:** Each arcade game costs up to $10, your allowance is $60, and you're also
supposed to save for dinner and the bus home. The per-game max and the allowance are
individually fine but jointly tight.

Probably intended (FM proxies were way under cap), but the doc should say what the
*expected* proxy spend is (~1–2 GPU-hr each), not just the ceiling, or wave-1 planning
will look infeasible on paper the first time someone does the arithmetic.

### 🟠 Critique 6: This is a research program squatting in the products repo

`templates/` in this repo = customer-facing console templates, each registered in
BUILD.yaml with CI and publishing. `autoresearch/` is none of those (not in BUILD.yaml,
no build, no test) — it's an internal research program wearing a template costume.

**ELI5:** It's a really good binder of lab notes... filed on the store shelf between
products for sale.

Maybe that's the plan (incubate here, productize later — and honestly "autoresearch as a
console template" would be a rad product). But right now the docs never say which one it
is. Two concrete risks: (a) repo tooling/pre-commit/CI conventions may eventually choke
on or flag an unregistered template dir; (b) paths like "clone the reference repo into
`~/anyscale/` as a sibling" bake *your laptop's* layout into docs whose core promise is
"any fresh session can resume from disk alone." The disk being the source of truth is a
great rule — but then the truth can't live in one person's home directory.

**Fix:** one paragraph in the README declaring intent (incubating here vs. destined for
its own repo), and swap `~/anyscale/...` sibling-clone instructions for a `$BASE`-style
env var like the campaigns already use for artifacts.

### 🟡 Critique 7: R3's canary aborts wave at the run from outside the fence

R3 says "the harness aborts the run the moment [a canary] fires." That implies an
external watcher streaming in-training metrics in near-real-time — log scraping (which
§9.11 of their own methodology says truncates silently!) or a metrics side-channel that
doesn't exist yet.

**Simpler:** the canary check lives *inside* the training loop as a callback — the run
checks its own vitals and exits nonzero with a canary-named exit message; the monitor
just records the reason. Self-destruct button beats a sniper watching through a window.
The FM month-canary was effectively this already. Doc should commit to the in-process
version; the external-abort version is a distributed-systems project nobody budgeted.

### 🟡 Critique 8: Pre-registration needs a "no editing history" rule

Seed plans as pre-registration is a genuinely great idea (it's what real clinical trials
do — you write the bet down *before* rolling, so you can't pretend you predicted
whatever happened, like calling your shot in pool). But it only works if the plan can't
be quietly edited after results come in. It's a git repo, so the rule is nearly free:
**amendments to a seed plan after the first paid run must be new commits with a visible
"AMENDED" note, never a rewrite** — and the campaign's registry rows should reference the
seed-plan commit SHA they ran under. One sentence in the template fixes it.

### 🟢 Things I'd defend if anyone pushes back on them

- **GPU-hours as the budget currency** — correct call, dollar prices are weather,
  GPU-hours are climate.
- **The two-gate reproduction rule** (their checkpoint + their eval first, then your
  pipeline) — this is the single best idea in the whole system. Never let anyone soften it.
- **Move-aside-never-delete (R8)** — paid for with a real $4 scar. Keep.
- **Fresh-context adversarial reviewers (R9)** with the "don't read the owner's notes"
  clause — this found the decisive flaw last time. (Fun meta-note: this file being *in*
  his working directory mildly violates that independence clause if he reads it mid-run.
  I'm the reviewer; he shouldn't. That's on us to enforce, not him.)
- **Eval pinning by content hash (R7)** — the "harness flip re-ranked the conditions"
  war story justifies every word.

### Watching for in the next commits

- The 9 seed plans + SEED_INDEX: are the "published numbers to match" pulled from eval
  *code*, or from paper abstracts? (Their own assumption #2 says code — easy to slip on.)
- Do wave assignments' cost estimates survive the Critique-5 arithmetic?
- Does any actual code show up, or does the doc pile keep growing? (Critique 1.)

---

## Round 2 — 2026-07-11, still uncommitted

**Reviewed:** `SEED_INDEX.md` + `campaigns/04-beir-dense-retrieval.md` (the first real
seed plan). Both new since Round 1.

**Credit first:** the BEIR plan is *properly* researched — he caught that the paper's
number (E5 v1, 50.0, 15 datasets) and the README's number (v2, 50.6, different subset)
don't match, flagged the Touché-2020 dataset revision, the "query:/passage:" prefix
footgun, the contamination critique of post-2023 models, and which datasets can't be
redistributed. This is exactly the level of paranoia the methodology asks for. The BM25
control (hypothesis-killer literature cited and all) is chef's-kiss.

Now the pokes:

### 🔴 Critique 9: The headline "beat" for campaign 04 is racing a guy on a bicycle with a truck

The primary beat thesis is cost/throughput: encode BEIR cheaper via Ray Data than the
reference's single-GPU + faiss loop. But **the reference never published a cost number**.
There's no $/1M-docs to reproduce, so there's nothing to *beat* — you'd be benchmarking
your fancy pipeline against a strawman you built yourself.

**ELI5:** "I beat the world record!" — "Whose record?" — "Mine, from this morning, when I
was hopping on one leg."

That's a perfectly good *platform demo* (and honestly the more useful artifact for
Anyscale marketing), but the program's whole brand is reproduce-then-beat *published
numbers*. Mixing the two dilutes the brand. **Fix:** either (a) relabel hypothesis 1
honestly as a platform-efficiency demo with a rigorously-defined baseline (same GPU
budget, naive-but-competent implementation, equal nDCG), or (b) make the quality beat
primary. Just don't let the blog post say "beat" without an asterisk the size of Texas.

### 🔴 Critique 10: The pooling hypothesis is probably dead on arrival — and it's predictable from mechanism

Hypothesis 2 ports the FM campaign's "readout thesis" (sweep readouts over frozen
embeddings, 7× swing possible!) to E5. But those situations aren't alike. The FM's
embeddings were pretrained generically — no one ever told them *how* they'd be read out,
so the readout was a free win waiting to be found. **E5 was contrastively trained
end-to-end THROUGH its mean-pooling.** The pooling isn't a hat you can swap; it's load-
bearing — the model's weights were optimized so that *mean-pooled* outputs land near
each other. Swap in last-token or attention pooling on frozen weights and you should
*expect* nDCG to drop.

**ELI5:** The FM was a buffet — how you fill your plate matters and nobody prescribed
it. E5 is a plated tasting menu — the chef already decided the arrangement, and
rearranging it in the kitchen doorway mostly makes it worse.

This is the first sign of the program's one real intellectual risk: **pattern-matching
the FM campaign's winning move onto domains where its mechanism doesn't hold.** The
portable-theses idea is good, but each port needs a one-line mechanism check: *why would
this work HERE?* Their own rule covers it — "probe before sweep" — so the fix is cheap:
hypothesis 2 gets ONE $2 probe (one dataset, one alt pooling) before any sweep, and the
seed plan should say so explicitly.

### 🟠 Critique 11: The proxy ladder conflates two different proxies

The plan's proxy rung is "the small BEIR sets" for ranking BOTH throughput ideas and
quality ideas. Quality: fine — small sets rank pooling/chunking changes. Throughput:
**broken** — encode throughput on SciFact (5K docs) tells you nothing about MSMARCO
(8.8M docs), because tiny corpora are all fixed overhead (model load, actor startup) and
zero steady-state. It's like timing a marathon runner over the first 40 meters.

**Fix:** two proxy axes, stated separately — quality proxy = small datasets; throughput
proxy = a fixed slice of the BIG corpus (e.g. 5% of MSMARCO), which is exactly the
"reduce the axis that scales cost while preserving the mechanism" rule his own template
prescribes. The plan's calibration note gestures at this but doesn't split them.

### 🟠 Critique 12: The 36GB elephant — this plan repeats the exact RAM trap the FM campaign fell into

MSMARCO is 8.8M docs × 1024-dim × fp32 = **~36GB of vectors**; add DBPedia (4.6M) and
exact top-k search over it all. The FM campaign needed a 128GB head node for a 3.55M×512
matrix (7GB!) because the eval path held everything in host RAM — it's cited as a war
story in their own budget doc. This plan's risk section talks dataset versions but
**never mentions the vector-matrix RAM budget**, and "faiss / exact, top-k" on the
driver is exactly how you rediscover it at 2am.

**Fix:** one line in §12: vectors stream to disk/object store, search is sharded (Ray
tasks over index shards, merge top-k) or faiss-on-GPU; head node stays small. They
already know this lesson — it just didn't make it into the plan.

### 🟡 Critique 13: Small bookkeeping wobbles in SEED_INDEX

- **Campaign 02's wave is self-contradictory:** the table says Wave "1→2", but the
  gating paragraph says "the 7B RL full run" is a *Wave 3* example (400–2,000 GPU-hr).
  Either the table or the paragraph is wrong; an approver reading only the table would
  green-light something the policy says needs Wave-3 sign-off.
- **First-wave spot math is high-shifted:** 120–160 GPU-hr at their own spot multiplier
  (×0.3–0.5 of ~$1/hr A10G) is ~$36–80, not "~$60–100." Within their ±50% religion, but
  the doc that preaches cost discipline shouldn't have its own arithmetic drift upward.
- Minor: campaigns 01–09 are promised as files; only 04 exists so far. Fine mid-flight,
  just tracking.

### 🟢 Round-2 things I'd defend

- Choosing **E5-large-v2 (2022) specifically to dodge benchmark contamination** — subtle,
  correct, and the kind of choice reviewers respect.
- **BM25 as the honest floor** with the Dacrema/RecSys-2019 receipts — most "neural beats
  X" papers die against tuned BM25; putting it in the controls section is real rigor.
- The **first-wave trio picks three different Ray workloads on purpose** (batch-embed,
  embed+probe, distributed-train) — that's the right way to stress-test the future
  harness rather than proving the same trick three times.

### Still watching for

- Seed plans 01, 02, 03, 05–09 (only 04 exists).
- Whether hypothesis-2-style "portable thesis" ports come with a mechanism check
  (Critique 10) or get pasted verbatim.
- Any actual code (Critique 1 stands).

---

## Round 3 — 2026-07-11, still uncommitted

**Reviewed:** `campaigns/05-prithvi-geospatial.md` + `campaigns/06-openvla-robotics.md`.

**Credit:** these are the two best-researched docs yet. 05 catches the deprecated repo
trap, the card-vs-paper number split (93.0 vs 88.6 — different protocols!), and the
contested-margin literature. 06 catches the unresolved GitHub issue where someone got 68%
vs the published 88.4%, the two 2025 papers showing LIBERO scores collapse under tiny
perturbations, and even the Llama-2 license buried in the weights. He's doing the
homework. Now the wobbles — and one of them is a genuine policy bug:

### 🔴 Critique 14: "GPU-hours" isn't one currency — an A100 hour costs 3.5 A10G hours, and the budget policy pretends they're equal

BUDGET_POLICY's whole pitch is "GPU-hours are the stable currency." But its wave
envelopes are stated in *raw* hours, tier unspecified — and the new seed plans quietly
break on this:

- **05 Prithvi:** ~35–55 GPU-hr total → that's *inside Wave 1's ≤60 cap*, yet it's
  assigned Wave 2. Why? Because the dollar line (~$120–200) assumes **A100** hours,
  which cost 3.5× A10G hours. The wave was assigned by dollars; the policy says hours.
- **06 OpenVLA:** ~120–220 GPU-hr → raw hours land in Wave 2 (60–400), yet it's Wave 3.
  Same reason: they're A100 hours (~$400–800), and he's *implicitly* weighting by tier.

**ELI5:** The rulebook says allowance is "60 coins." But some coins are pennies (T4) and
some are half-dollars (A100), and the kids have started counting in whichever coin makes
their plan look right.

His instinct (weight by tier) is correct — it's the policy that's underspecified. **Fix:**
one line in BUDGET_POLICY: *envelopes and rung caps are in A10G-equivalent hours,
weighted by the rel-cost column already sitting in the conversion table* (T4 ×0.5, A100
×3.5, H100 ×5.5). Then 05 is legitimately Wave 2 (~120–190 A10G-eq) and 06 legitimately
Wave 3 (~420–770 A10G-eq) — the waves he picked become *derivable* instead of vibes. The
registry's `cost` field should store both raw and A10G-eq. Cheap fix, and without it the
R4 submit-time cap can be gamed by tier choice.

### 🟠 Critique 15: The program has quietly stopped being "beat the number" — and that's good, but the front door still says otherwise

Look at the three seed plans' actual theses:
- 04 BEIR: beat on **cost** (no published number to beat — see Critique 9).
- 05 Prithvi: an **honest reckoning** — "measure whether the FM's advantage over a tuned
  U-Net is even real (it may be small)."
- 06 OpenVLA: explicitly **refuses** to play beat-the-number ("commoditized") and makes
  the *finding* — the robustness gap — the contribution.

Zero of three are a straight "beat the published metric." Meanwhile README.md line 1
still brands the whole program "benchmark-anchored ML hill-climbing" / reproduce-then-
**beat**. This drift is the *right* science — "the truth is the deliverable" beats
leaderboard-chasing — but if the front door promises hill-climbing and the campaigns
deliver "actually the published margin is mostly noise," that reads as failure to anyone
(a boss, a budget approver) who only read the front door.

**ELI5:** The sign on the shop says "We beat any price!" but inside, the actual product
is really careful consumer reports. Great product! Wrong sign.

**Fix:** a short "what counts as a win" section in README: (a) metric beat, (b) cost/
efficiency beat at held quality, (c) a rigor finding that revises the published claim
(robustness gap, honest-baseline reckoning). All three ship a blog post. Then 04/05/06
are on-brand instead of quietly off-mission.

### 🟠 Critique 16: 06's "embarrassingly parallel" sim fleet has a 7B elephant in every worker

The plan's coolest Ray story — "hundreds of MuJoCo workers, embarrassingly parallel
rollout eval" — glosses over where the **7B model forward pass** happens. A LIBERO
rollout is a *loop*: sim step (CPU) → policy inference (GPU) → sim step → ... If each of
500 parallel workers owns its own 7B-loaded GPU, the GPUs sit ~idle waiting on CPU sim
steps and the bill explodes; that's not embarrassingly parallel, it's embarrassingly
*expensive*.

The right architecture (and the actually novel Anyscale story) is **decoupling**: a CPU
fleet of sim workers streaming observations to a small pool of *batched* GPU inference
actors (their own fractional-GPU/continuous-batching instincts, pointed at robotics).
The plan's budget (~10–20 GPU-hr for the eval gate) is only achievable under that
design, but §7 just says "Ray Tasks / actor fleet" — the one sentence that decides
whether the estimate is real is missing. Add it to §7 and §12.

### 🟡 Critique 17: Smaller notes on 05

- The gate offers two published numbers (card 93.0, cheap; paper 88.6, needs ~20 runs)
  and says "pre-register which" — good, but the *budget* only works for the card number.
  Chasing the paper's 10-seed × 10-trial HPO protocol is a different campaign. Just
  pre-register the card number now and say the paper protocol is out of scope.
- Hypothesis 2 (tune the U-Net baseline properly) is the highest-value thing in the
  whole plan and arguably *the* deliverable — consider promoting it from "hypothesis"
  to the campaign's stated thesis, like 06 did with robustness. The contested-margin
  literature he cites basically predicts the outcome; own it up front.

### 🟢 Round-3 things I'd defend

- **06 refusing the commoditized beat** ("OFT is at 97%, don't chase 76.5%") — this is
  taste. Most people would have chased the stale number.
- **05's scene-level holdout warning** (never random tiles from the same scene) — the
  imagery version of temporal leakage, stated exactly right.
- **06's random-action control** (success ≈ 0 or your harness is lying) — cheap, decisive.
- Licensing flags in both (Llama-2 community license on OpenVLA weights; per-dataset
  redistribution in BEIR) — nobody does this and everybody should.

### Still watching for

- Seed plans 01, 02, 03, 07, 08, 09 (4/10 exist: 04, 05, 06 + index).
- Whether BUDGET_POLICY gets the A10G-equivalent fix (Critique 14) or waves stay vibes.
- Still zero code, zero commits. The doc pile is now ~8 files. (Critique 1 compounds.)

---

## Round 4 — 2026-07-11, still uncommitted — THE WHOLE SET IS NOW VISIBLE

**Reviewed:** the last four campaign files — `00-transaction-fm-fraud` (the worked
example), `01-hstu-generative-recsys`, `02-rl-reasoning-zero`, `03-dlrm-ctr-mlperf`. All
10 campaigns + index + policy + requirements + readme + template now exist (16 files).

**Big credit:** Two standouts. **00** is a genuinely excellent worked example — every
schema field filled with a *real* finished answer (the shuffled-label AP-collapse-to-
0.0016, the velocity baseline at 0.076-below-raw, Zach's faithful replication landing 4×
below). Anyone starting campaign #2 can see exactly what "done" looks like. And **03**
catches the single most important landmine in the whole program: Criteo is CC-BY-NC
(non-commercial), which *directly conflicts with the program's stated marketing purpose*,
and he escalates it as a pre-registration PI decision instead of discovering it at
publish time. That's the methodology working.

Now that I can see all ten at once, three of my earlier one-off critiques turn out to be
*systematic*, and that's the real story of this round.

### 🔴 Critique 18: The "cost beat" isn't one weak hypothesis — it's the primary thesis of THREE campaigns, and all three beat a strawman

Critique 9 (BEIR) is not a one-off. The **primary** beat in 04 (BEIR), 07 (Chronos), and
09 (Whisper) is all the same move: "encode/forecast/transcribe cheaper via Ray Data than
the reference's naive single-GPU loop." In **every** case the reference published *no
cost number*. So the thing being "beaten" is a baseline the campaign builds itself — and
whoever writes the pipeline decides how slow the strawman is.

**ELI5:** Three of the nine kids are entering the "who's fastest" race, and in all three
the only other runner is a scarecrow they propped up at the start line.

This is fine *engineering* (Ray Data batch inference genuinely is cheaper, and it's
Anyscale's best story) but it is not *reproduce-then-beat*, and having it be the headline
of a third of the program is a brand problem. **Fix (program-level, one place):** define
a "cost/efficiency beat" as its own claim type with a *mandatory rigor rule* — the
baseline must be "a competent single-GPU implementation given the same hardware budget,
tuned to within reason," not the reference's demo script, and nDCG/WQL/WER must be held
equal within CI. Put that rule once in REQUIREMENTS or BUDGET_POLICY so all three
campaigns inherit it, rather than restating (or forgetting) it three times.

### 🔴 Critique 19: The program is built for marketing, but "can we legally market this?" is not a gate anywhere

03's Criteo catch exposes a hole. The README says vertical/marketing fit drives which
campaigns ship. Yet look at the license landmines scattered across the set: Criteo
CC-BY-NC (03, blocks marketing), OpenVLA weights under Llama-2's >700M-MAU clause (06),
ESM-3 non-commercial (08, dodged by using ESM-2), MovieLens research-use-only (01). He
caught each *individually, by hand, per campaign* — exactly the "informal human-in-the-
loop" the whole program is supposed to mechanize away.

**ELI5:** You're opening a bakery whose entire point is selling cakes, but "is this
ingredient legal to sell?" is a thing each baker is trusted to remember, not a checklist
at the door. One forgotten nut allergen and you can't sell the cake you spent the GPU
budget baking.

**Fix:** add a first-class **`commercial_use: yes/no/needs-legal`** field to the seed
template §3, and make it a pre-registration gate in REQUIREMENTS alongside the
reproduction gate — *no campaign whose output can't be used for its stated purpose spends
budget until the PI signs the license risk.* This is cheap (one field + one gate) and it
turns four hand-catches into one enforced rule.

### 🔴 Critique 20: Campaign 02 is mis-waved by a full tier — and it's the most expensive run in the program

02 (zero-RL) says "Wave 1 for proxy, Wave 2 for the 7B full run," then in the same
breath: "a single full run is ~120 GPU-hr... H100... ~$400–700 on spot, multi-node." Run
that through Critique 14's tier weighting: 120 **H100** hours × 5.5 = **660 A10G-equiv
hours** — that's not Wave 2 (60–400), it's solidly **Wave 3**. By dollars ($400–700/run,
multi-node) it's also Wave 3. The doc even lists the Wave-3 triggers (multi-node, each
full run) in its own approval line — but labels the wave "2."

This matters more than the 05/06 wobbles because 02 is *the budget-dominant campaign* and
an approver skimming "Wave 2" would apply the wrong (lighter) sign-off ritual to the
single most expensive, most-likely-to-run-away RL job in the set. It's the strongest
possible evidence for Critique 14: **without tier-weighting, the wave label is decorative,
and the one time it's most dangerous to be wrong, it's wrong.**

### 🟠 Critique 21: The "readout thesis" is being ported to 5 campaigns, and its mechanism only actually holds in ~2 of them

The portable readout thesis (sweep the readout on frozen embeddings, find a big swing)
now appears in 01, 04, 05, 08, and echoes in 07/09's "decoding" hypotheses. But whether
it *works* depends entirely on whether the reference model was trained end-to-end through
its readout — and that varies:

| Campaign | Readout thesis will... | Why (mechanism) |
|---|---|---|
| **08 ESM-2** | **work — it's the whole point** | frozen backbone, readout was never jointly trained; FLIP lit *already shows* 0.54→0.79. This is the honest port. |
| 01 HSTU | probably work | trained from scratch by us; readout genuinely up for grabs |
| 05 Prithvi | maybe (decoder, not readout) | segmentation head swap is a real question but not "free" |
| **04 E5** | **probably fail (predictable)** | contrastively trained *through* mean-pooling — see Critique 10 |
| 07/09 decode | different thing entirely | decoding params ≠ readout; don't conflate |

**ELI5:** "How you serve the dish" only matters if the chef didn't already plate it.
08's chef handed you raw ingredients (works great). 04's chef already plated it (swapping
mostly ruins it). Same move, opposite outcome, and it's knowable in advance.

08 is the campaign to lead with precisely because it's the *honest* test of the thesis —
and the SEED_INDEX already ranks it #2 in wave 1, good. **Fix:** make the seed template's
hypothesis section require a one-line **mechanism check** for every ported thesis ("why
would this work *here*, given how the reference was trained?"). His own "probe before
sweep" rule already implies it; the template should force it onto the page so a weak port
(04) can't masquerade as a strong one (08).

### 💡 The synthesis I'd put in front of you, boss

Read all ten and a pattern jumps out that's bigger than any single critique. **Look at
what the actual deliverable of each campaign is:**

- 02: "the base-model floor is honest, the aha-moment is a reward artifact"
- 03: "the audited gate is hard; AUC gains past it are often illusory"
- 05: "the FM's margin over a tuned U-Net is small — here's the honest number"
- 06: ">90% success collapses toward 0 under tiny perturbations"
- 07: "classical baselines beat the neural forecaster under fair eval"
- 04/09: "the cost win is real; the quality is held flat"

**Almost none of these is "we beat the leaderboard."** The real, repeated product is
*rigorous re-evaluation that corrects an overstated published claim.* That is a **better
and more defensible product** than benchmark-chasing — it's the thing the FM campaign
actually did (their embedding beat their fusion, and their reference contradicted its own
blog). But the front door (README line 1: "benchmark-anchored hill-climbing... beat")
sells the weaker story.

If this were mine, I'd rename the whole program's pitch from *"reproduce then beat"* to
something like *"reproduce, then find the truth the leaderboard hides."* Keep every Iron
Rule — they're what make the re-evaluations credible. Just stop promising hill-climbing
when the actual, more valuable output is honest forensics. (This is Critique 15,
confirmed and hardened now that all ten agree.)

### 🟢 Round-4 things I'd defend hard

- **00 as the worked example** — filling every schema field with a real finished answer
  is the single best onboarding artifact here. Keep it permanently, never let it rot.
- **03's license escalation** — textbook. It's the seed for Critique 19.
- **02 using contested-claim literature to pre-emptively disarm its own hype** ("don't
  headline the aha-moment; oat-zero showed it's at epoch 0") — this is the discipline
  most researchers lack.
- **01 leading with Amazon Books, not MovieLens** because ML is saturated — resisting the
  easy-but-unpersuasive number is exactly right.
- Every campaign names its **honest cheap floor** (BM25 / Seasonal-Naive+AutoARIMA /
  tuned U-Net / most-popular / logistic-regression) instead of the weakest quoted
  baseline. That single habit is what separates this from 90% of ML blog posts.

### Verdict at the end of the doc phase

The research quality is high and remarkably consistent — every plan does real
homework, cites contested claims, names honest floors, and flags licenses. My four 🔴s
are **not** "this is wrong," they're "these local good instincts should be promoted to
enforced program rules before they get forgotten at scale": (18) cost-beat rigor,
(19) commercial-license gate, (20)+(14) tier-weighted waves, (21) mechanism-check on
ported theses. All four are one-paragraph fixes.

The one strategic move I'd actually push on: **the program's stated mission undersells
its real product** (the synthesis above). Worth a real conversation.

### Still watching / open threads

- **Critique 1 is now the loudest:** 16 files of docs, still **zero commits, zero code.**
  The plan is impeccably specified and entirely unexecuted. The next artifact that would
  change my assessment is `registry.py`, not doc #17.
- Does BUDGET_POLICY get the A10G-equivalent-hours fix? (14 + 20 both depend on it.)
- Do the four "promote to a rule" fixes land, or stay as per-campaign hand-catches?

---

## Round 5 — 2026-07-11, FIRST COMMIT LANDED (`56d9896c`) + scorecard

He committed all 15 docs as one commit (`autoresearch: program docs + 9 pre-registered
seed campaigns`) — and correctly left this `critiques.md` **out** of it (stays untracked).
Before committing he revised the core docs, and **he's clearly been reading this file**:
several fixes landed in near-identical language (the mechanism-check even reuses my exact
E5-mean-pooling example). Scorecard of everything I raised:

| # | Critique | Status in commit |
|---|---|---|
| 1 | Docs-before-code; guard against n=1 over-building | ✅ **LANDED** — REQUIREMENTS "Guard against over-building": build only R1+R2, let campaign #2 rewrite |
| 5 | Proxy cap vs Wave-1 envelope arithmetic | ✅ **LANDED** — BUDGET_POLICY: "expect ~1–2", ceiling ≠ expected spend |
| 6 | Research program squatting in products repo + laptop paths | ✅ **LANDED** — README "Where this lives" + `$AUTORESEARCH_REFS` |
| 8 | Pre-registration needs a no-silent-edit rule | ✅ **LANDED** — README step 1: frozen after first paid run, `AMENDED` commits, SHA in registry rows |
| 21 | Mechanism-check on ported theses | ✅ **LANDED** — `_TEMPLATE` §9: mandatory mechanism check (uses my E5 example) |
| 14/20 | Tier-weighted (A10G-equiv) wave accounting | ❌ **OPEN** — BUDGET_POLICY still counts raw hours; 02/05/06 still mis-waved |
| 18 | "Cost beat" rigor rule (3 campaigns beat a strawman) | ❌ **OPEN** — no efficiency-beat claim type defined |
| 19 | Commercial-use license gate (program is for marketing!) | ❌ **OPEN** — no `commercial_use` field / gate; still 4 hand-catches |
| 3,4 | R1 machine-readable decisions / who-writes-the-row | ⏳ unverified (R1 section changed some; didn't re-diff line-by-line) |
| 9–13,16,17 | BEIR/campaign-specific | ⏳ campaign files committed at the versions I reviewed — not revised |

**Read the split:** the 5 that landed are all *"add a paragraph"* fixes. The 3 still open
(14/20, 18, 19) are the ones that need a *structural* edit to BUDGET_POLICY / REQUIREMENTS
— define a new claim type, add a weighting column, add a gated field. Could be he ran out
of runway before committing, could be disagreement, could be just-not-yet. **These three
are the highest-value remaining doc work** and I'd push on them before any code:
- **19 (license gate)** is the scariest — the program *exists to make marketing* and
  "may we legally market this output?" is still trusted to per-campaign memory. One miss =
  budget spent on an unshippable result. Cheapest possible fix, highest downside avoided.
- **14/20 (tier weighting)** — until this lands, the 7B RL run (the program's single most
  expensive job) wears a "Wave 2" label that triggers the wrong, lighter approval ritual.
- **18 (cost-beat rigor)** — a third of the program headlines a beat with no published
  number to beat; define the honest-baseline rule once, centrally.

### 🟠 Critique 22 (meta, for the PI): the reviewer and the author are sharing a scratchpad

This is the interesting one. The single highest-leverage move in the whole methodology
(R9, and the FM campaign's own war story) is the **fresh-context, independent adversarial
reviewer** — the one carrying the clause "do NOT anchor on the owner's notes; assume it's
wrong until you can't break it." That independence is *load-bearing*: it's what found the
decisive flaw last time.

But right now, the author is reading the reviewer's file mid-run and adopting its framing
(sometimes verbatim). That's *great for velocity* — good ideas land in minutes — but it
quietly **collapses the exact independence the program says is its best asset.** The five
landed fixes weren't independently re-derived and stress-tested by him; he took my word
for them. If my reasoning on one was wrong, it's now baked in with no second pair of eyes.

**ELI5:** The whole point of a second referee is that they *didn't* see the first
referee's scorecard. If they're both filling in the same sheet, you don't have two
referees — you have one with an echo.

Not a "stop" — the ideas are sound and landing fast is good. But for the *real* campaigns,
the R9 reviewer must run in a **fresh context that never sees the author's working notes
or this file**, or the independence guarantee is theater. Worth making explicit in
REQUIREMENTS R9: the adversarial agent's prompt is code + data + literature only, and
critiques flow back through a gate (the PI), not a shared live document. (I'm mildly
guilty of enabling this by putting the file in his working dir — that's on the setup, not
him.)

### Verdict after the commit

Genuinely encouraging. He takes review well and moves fast — 5 fixes in one commit, the
worked example is preserved, the repo is clean, nothing over-committed. The three open
structural fixes (18/19/14) are what I'd want to see in commit #2. And still — **zero
lines of code.** The n=1 guard rule he just adopted (my Critique 1) says the next move is
`registry.py`, not more docs. Watching for whichever comes first.

### Watching for next

- A follow-up commit addressing 18 / 19 / 14 (the structural three).
- The first actual code (`registry.py` / a launcher) — his own newly-committed rule says
  this is the next step.
- Whether R9 gets the independence-hardening from Critique 22.
- If commits go quiet from here: I take the lead and draft the minimal R1 `registry.py` +
  the three open doc fixes myself.

---

## Round 6 — 2026-07-11, author went quiet → I took the lead

He committed once (`56d9896c`) and stopped — ~55 min quiet across two heartbeats, HEAD
unchanged, no code. Per the standing instruction ("if commits stop, take the lead"), I
built the thing his own newly-adopted rule (#1) says comes next: the first real **code**.

**Scorecard update — he addressed MORE than Round 5 showed.** Re-reading the revised
REQUIREMENTS, Critiques **#3 and #4 also LANDED** (typed `decision` field + `seed_plan_commit`
+ the single-writer/idempotent-on-(run_id,status)/monitor-writes-terminal design, almost
verbatim). Tally of everything raised: **#1, #3, #4, #5, #6, #8, #21 landed in the docs.**
Still open: **#14/#20 (tier weighting), #18 (cost-beat rigor), #19 (license gate)** + the
campaign-specific ones (#9–13, #16, #17), which are per-campaign not program-wide.

### What I built (additive — did NOT touch his authored docs)

- **`harness/registry.py`** — R1 reference implementation, ~180 lines, stdlib only. Schema
  verbatim from REQUIREMENTS; single new thing = tier-weighted `a10g_equiv_hours` in cost.
- **`harness/test_registry.py`** — 11 tests, **all passing**, each an executable
  REQUIREMENTS clause or a critique-made-concrete. Verified end-to-end via the CLI: a
  two-campaign registry reconstructs full state (spend, promotes, kills-with-reasons,
  eval-pins) from JSONL alone — the R1 "Done when" acceptance check, green.
- **`PROPOSALS.md`** — the three open fixes as PI-decidable proposals (P2 tier-weighting
  now demonstrated in code; P3 license gate; P4 cost-beat rigor).

### Why code, and why additive

Critique #22 says the reviewer must not blend into the author's work. So rather than rewrite
his committed `.md` files, I put my contribution in a new `harness/` dir + `PROPOSALS.md`,
clearly mine. If he comes back, zero collision on his files.

The strongest result: **#14 is no longer an argument, it's a passing test.**
`test_the_7b_rl_run_is_miswaved` asserts 120 H100-hr = 660 A10G-equiv > 400 → the "Wave 2"
label on the program's most expensive run is provably wrong. That's a better artifact than
another paragraph of critique.

### Note for the PI

The monitor now fires on **my own** file writes too (it can't tell author from reviewer on
a shared branch) — so from here, "new file" events may be mine. The meaningful signal for
*his* return is a new **commit** (I don't commit; my work sits as untracked working-tree
files for you/him to review).

### Watching for next

- Author returning (a new commit) — I'll review it and stop leading.
- Otherwise: R2 (launcher) is the next code per build-order, but it needs real Anyscale
  job-submit surface — I'd want your go-ahead before speccing that against live infra.

### Round 6b — adversarial self-review of my own R1

Applied the program's own R9 medicine to `registry.py` before leaving it as "the" R1
suggestion. One real latent gap: idempotency dedupes exact `(run_id, status)` pairs but
can't catch a run with two *conflicting* terminal statuses (a buggy monitor writing both
`FAILED` and `SUCCEEDED`) — it would silently let the last writer win. Fixed in the same
spirit as the cross-pin check: `conflicting_terminal_runs()` **surfaces** it rather than
silently resolving it, and `reconstruct()` now reports it. Test added; **12 passing**.

Boundary reached. Everything left is user-side: (a) the author committing something new
(I resume review), or (b) PI decisions on `PROPOSALS.md` P3/P4 and a go-ahead on R2
(touches live Anyscale infra — I won't spec it against real job-submit unprompted).
Holding in watch-only mode; the monitor wakes me on any real commit from him.

---

## Round 7 — 2026-07-11, PI said "fix everything + build the viz" → applied + shipped

The PI took me off watch-only and told me to apply the fixes (trusting my gut/research over
the author's) and build a visualization. Meanwhile the **author came back and is editing the
working tree in parallel** (no commits yet) — he independently applied the campaign-04
critiques (#9/#10/#11/#12), using my own analogies verbatim ("marathoner over 40 meters").
So I split the work to avoid collision: I took the **program-level** docs and the viz; his
campaign-file edits stand where they already cover the critique.

### Fixes applied (program-level, by me)

- **BUDGET_POLICY** — A10G-equivalent hours is now the stated currency; wave table relabeled;
  added the campaign-02 worked example (120 H100-hr → 660 A10G-eq → Wave 3). (#14/#20)
- **REQUIREMENTS** — two new non-functional gates: #7 commercial-use license gate (with the
  four real landmines named), #8 cost/efficiency-beat as a distinct claim type with the
  fair-baseline rule. (#18/#19)
- **_TEMPLATE** — `commercial_use` field in §3; cost-beat honesty note in §9; §10 budget now
  carries an A10G-equivalent total that sets the wave. (#18/#19/#14)
- **README** — retitled and reframed to "reproduce, then find the truth the leaderboard
  hides"; added a "What counts as a win" section (metric / efficiency / rigor-finding). (#15)
- **Campaign 02** — Wave 2 → **Wave 3** corrected with the A10G-eq math shown. (#20)
- **Campaign 06** — added the sim-fleet decoupling note (CPU sim workers → batched GPU policy
  actors; a 7B-per-worker layout would blow the budget). (#16)
- Campaign 04's fixes were done by the author himself (#9/#10/#11/#12) — left as-is.

### Shipped: the live visualization

`viz/mission-control.html` — a monospace telemetry console (published as an Artifact).
Ten campaigns flow through the Ray pipeline spine (Source → Preprocess → Repro-Gate → GPU →
Registry → Eval → Serve); a temporal scrubber drives run-pulses, budget meters in
A10G-equivalent hours (campaign 02's bar visibly overruns the Wave-2 cap line — the fix, made
visual), a hypothesis tree that resolves promote/kill/hold as time plays, and hover-for-open-
questions on every stage. Data model = `harness/registry.py`. Verified: no JS errors, DOM
correct, temporal engine drives log/budgets/tree (34 registry rows at T=78%, outcomes match
the critiques — 08 readout promotes, 04 pooling kills, 06 robustness-finding promotes).

Everything sits as untracked working-tree files + one published Artifact. I still don't
commit — the author/PI owns what lands on the branch.

---

## Round 8 — 2026-07-11, harness expanded (R4/R7/R8) + tests

PI said "keep building anything else." Extended the harness with the pieces the build-order
names as R1-adjacent and that need **no live infra** — turning three more prose requirements /
critiques into tested code (same move as R1 → tier-weighting-as-a-passing-test):

- **`budget.py` (R4)** — closes critique #2 ("prepaid card, not a pinky promise").
  `estimate()` tier-weights cost; `preflight()` refuses over-cap / over-envelope runs and
  escalates `full` to the PI (reading real spend from the registry); `wall_clock_timeout_s()`
  is the runtime hard-kill. A test proves campaign 02's 660 A10G-eq run is refused under a
  Wave-2 envelope.
- **`artifacts.py` (R8)** — the never-delete / move-aside primitive (my #1 memory, the $4
  deletion). Idempotent + guarded: a double-submit never clobbers a backup.
- **`evalpin.py` (R7 part)** — content-hash an eval spec/file → the `eval_pin` every row
  carries; order-independent, any content change → new pin.
- **`harness/README.md`** — navigable index + how-a-campaign-uses-it.

**Verified: 27 tests green** (registry 12 · budget 6 · artifacts 5 · evalpin 4) + an
integrated smoke running estimate → preflight → registry → spend → timeout → eval_pin
end-to-end. One test bug caught + fixed in the loop (I'd labeled an A100 estimate 660; A100 is
×3.5 = 420, H100 is ×5.5 = 660 — the code was right).

Deliberately still NOT built (need a 2nd campaign or PI sign-off to justify): R2's job-submit
(live infra), R3 canaries, R5 proxy-calibration, R9 review-agents. Not over-building from n=1.

Author status: quiet since his campaign-04 working-tree edits; HEAD still `56d9896c`; the
`M` files in `git status` are all my program-level fixes. No collisions.

---

## Round 9 — 2026-07-11, viz gets a LIVE mode (closes the registry→dashboard loop)

Author still quiet; PI said keep building. Upgraded `viz/mission-control.html` from demo-only
to a real tool: a **"load registry ▾"** drawer accepts JSONL straight from `harness/registry.py`
(prefilled with a runnable sample), parses runs + typed decisions, and drives the **budget
meters and registry log from real rows** — same aggregation as `registry.campaign_spend`
(terminal rows only, deduped by run_id). Demo mode (the animated timeline) stays the default
and untouched; a toggle flips between them.

The payoff: paste real data and campaign 02's meter shows **660 A10G-eq, amber, over the
Wave-2 cap line** — the tier-weighting fix, now provable from an actual ledger, not a
scripted demo. Republished to the same Artifact URL.

Verified in-browser: no JS errors (only the harmless favicon 404); go-live shows 4 runs / 2
decisions with the promote (08 readout) + kill (04 pooling) rows; back-to-demo restores and
resumes the clock cleanly; play/scrub correctly disabled in live mode.

---

## Round 10 — 2026-07-11, R2 launcher (spec-gen, up to the PI boundary)

Author still quiet. Built `harness/launcher.py` — the R2 "one command per experiment" spec
generator, the pure part that needs no live infra:

- `build_job_spec(exp, base)` runs the R4 budget preflight first (refuses over-cap/over-
  envelope, marks `full` runs approval-required), then bakes in every `BUDGET_POLICY` default:
  spot + `fallback_to_on_demand`, `min_nodes:0` scale-to-zero, `resources:{CPU:0}` GPU fence,
  on-demand stateful head, wall-clock timeout from the estimate, greppable run name (Iron
  Rule #5), flags default-OFF (Iron Rule #8), eval-pin + registry path threaded into the
  entrypoint.
- `submit()` is **the escalation boundary** — returns the CLI it *would* run, refuses `full`
  without `pi_approved=True`, and **never calls Anyscale**. Wiring the real submit is a
  separately-approved step.

This ties the whole harness together: estimate → preflight → spec → (boundary) → registry →
artifacts → eval-pin. **33 tests green** now (added 6 for the launcher); end-to-end demo
generates campaign 02's full-run spec, flags it approval-required at 660 A10G-eq, refuses
submit without sign-off, and returns the CLI with it.

Harness coverage now: **R1 ✓ R2 ✓(spec) R4 ✓ R7 ✓ R8 ✓**. Deliberately unbuilt: R2's live
submit (PI), R3 canaries, R5 proxy-calibration, R6 monitor (needs live job state), R9 review
agents — each needs a real campaign or PI sign-off to justify its shape. That's the honest
edge of what's buildable without touching infra.

---

## Round 11 — 2026-07-11, adversarial self-audit (R9 on my own code) → found a real bug

Rather than manufacture more speculative code past the infra-free edge, applied the program's
own highest-leverage move — the fresh-eyes adversarial review — to *my own harness*. It paid
off: found a **real safety bug** in `launcher.py`.

**The bug:** `approval_required = bool(verdict and verdict["escalate"])`. When `build_job_spec`
was called **without** a registry `base` (no spend context), `verdict` is `None`, so a `full`
run came back `approval_required=False` — and `submit()` would have let a baseless full run
through with no PI sign-off. Crossing into `full` is a *policy fact*, independent of any spend
check, so it must always require approval. Fixed:
`needs_approval = (rung == "full") or bool(verdict and verdict["escalate"])`.
Regression test added (full run, `base=None`, still refused by `submit()`).

Two correctness nits fixed too: `_flag_args`/`run_name` now omit a flag set `False`/`None`
(Iron Rule #8 "default OFF" — previously would have emitted `--flag False`).

**35 tests green** (launcher 6→8). This is the kind of bug the whole methodology exists to
catch — and it was in *my* code, found by treating it as guilty until proven correct. Good
reminder that the reviewer isn't exempt from review.

---

## Round 12 — 2026-07-12, the CRUX: the "beat it" / incremental-improvement engine

PI reframed the whole repo: reproduction is table-stakes (and often skipped — many users bring
their own model + baseline); **the actual product is a rigorous, structured machine for beating
a number over and over and compounding wins.** The prior work was all *defense* (don't fool
yourself); this is the *offense* that was missing. Built:

- **`BEAT_IT.md`** — the incremental-improvement engine, profoundly + playfully. Reframes the
  gate to "trusted baseline + trusted eval" (three ways in, incl. bring-your-own-model). The
  loop: MEASURE → DIAGNOSE → HYPOTHESIZE → TEST → READ → DECIDE/BANK. Key sections:
  - **DIAGNOSE = error analysis as the engine of direction** — slice the eval, ceiling/oracle
    analysis, and the **diagnosis→direction table** (symptom → cause → cheapest test → next
    move). This is the "interpretation → what test next" brain the PI asked for.
  - **The downstream-probe pattern** (the PI's explicit point) — the FM→classifier trick
    generalized: declare a cheap probe as the thermometer for the expensive thing; a probe
    ladder (linear → MLP → light-FT → full); extract-once/sweep-many is exactly where the
    fraud campaign's 7× readout lift lived.
  - **Anti-fooling-yourself-while-climbing:** noise floor per decision, two-eval discipline
    (dev to climb / holdout as a spent budget — the reusable-holdout idea), confound firewall
    (one change), regression re-check, multiple-comparisons + power awareness.
  - **SEE IT:** climb chart, marginal-lift waterfall, per-slice heatmap, the climb *tree* —
    all reads over the R1 registry + the dashboard.
  - A worked example casting the fraud FM campaign *as a climb* (the readout sweep is the star,
    reproduction just certified the altimeter).
- **`harness/erroranalysis.py`** — the DIAGNOSE step as tested code: slice a scored eval, rank
  worst/biggest slices, and emit **ranked hypothesis cards** (lift-per-GPU-hour) from the
  diagnosis table. Metric-agnostic. **7 tests; 42 total green.** Demo on a fraud-shaped eval
  ranks "fix the burst slice" (smoke-cost, high lift) first, then the overfit fix, then flags
  the within-noise gain and the not-earning-its-keep baseline as warnings.

Designed-but-not-built (need a real 2nd campaign to shape, per the anti-over-building rule):
`recipe.py` (current-best stack + marginal-lift + regression guard), `probe.py` (the probe
ladder), and the climb chart/waterfall/tree dashboard views. Specs live in `BEAT_IT.md`.

---

## Round 13 — 2026-07-12, completed the end-to-end engine + stress-tested by simulation

PI pushed: it wasn't fleshed out end-to-end, and asked me to (a) build the missing tools so it
can hill-climb ANY ml job, (b) stress-test by simulating runs and fixing what breaks, (c) do a
lit review. All three done.

**Built the missing spine (all tested):** `metrics.py` (bootstrap CI + paired-bootstrap
significance — the noise floor), `contract.py` (the any-job eval-output contract + confound
firewall `single_delta`), `recipe.py` (the compounding stack: marginal-lift, regression guard,
dev/holdout budget), `climb.py` (the controller — `propose_next` drives one loop turn end to
end, up to the submit boundary). **68 tests green.**

**Simulation-driven bug fixes (the valuable part — unit tests passed, composition broke):**
1. `recipe.best_score` returned the *last* trick's score → a single non-improving "dud" trick
   silently lowered the running best, corrupting every later marginal. Fixed to track the true
   running best; regression test added.
2. **Slicing a ranking metric (AP) is degenerate.** An end-to-end sim showed per-slice AP on
   label-correlated slices is meaningless (all-positive slice → 1.0, all-negative → 0.0), so the
   diagnosis engine flagged the *negative class* as "the worst slice, 80% of eval" and made it
   the top recommendation — then crashed on the empty plan. Fixes: (a) `min_pos` guard excludes
   no-positive slices from direction; (b) new `recall_by_subgroup` scores subgroups against the
   *global* ranking (the correct ranking error-analysis); (c) `diagnose` now compares worst-vs-
   best subgroup *within one metric space* instead of against a possibly-different overall
   metric. Three regression tests added.
3. Empty-input guards on the bootstrap fns (an empty slice would crash on `randrange(0)`).
4. Documented AP's score-tie order-dependence.
The composed mini-campaign now runs clean: baseline AP 0.62 → subgroup recall localizes the
weak "burst" frauds → readout fix lands a CI-separated +0.38 → banked. This is the whole loop.

---

## Round 14 — 2026-07-13, nasty sims + the rigor essentials from the lit review

PI wants a first real run today; asked for demanding simulations + implement all the missing
lit-review essentials + slice auto-discovery + dashboard. Done.

**Two nasty simulations, each caught a real problem (one was a genuine bug):**
- SIM A (multiple comparisons): 40 NULL experiments → 5 "passed" the per-test noise floor.
  Investigating exposed a **bug**: `is_real_gain` was non-directional (CI excludes 0 in *either*
  direction), so it counted reliable *regressions* as wins — all 5 had negative deltas. Fixed to
  directional (`lo>0` for higher-better; `hi<0` for WER). Re-run: **0 false wins**, and BH-FDR
  across the campaign confirms 0 significant. Napkin: expected_false_wins(40)=2.
- SIM B (adaptive overfitting): keeping best-of-80-on-dev over noise-only candidates inflated dev
  +0.12 over truth. Thresholdout now flags `overfit=True` and returns the honest holdout (0.28)
  instead of the illusory dev (0.46).

**Built the lit-review essentials (all tested):**
- `significance.py` — Benjamini-Hochberg FDR + Bonferroni (the multiple-comparisons gap).
- `holdout.py` — `Thresholdout` (Dwork; two-eval discipline with teeth).
- `metrics.py` +`p_value_gain` (feeds FDR), +`min_detectable_effect`/`detectable` (power up
  front — replaces the `positives_warning` stub), directional `is_real_gain` fix.
- `erroranalysis.py` +`discover_slices` — auto-find the weak subgroup unprompted (SliceLine-style,
  ranking-aware via global-cut recall).
- `canary.py` — R3 leak audit (shuffled-label control, too-good-too-early, embedding collapse).
- `viz/climb.html` — **the dashboard climb view** (new Artifact): metric-vs-experiment with the
  noise band + promote/kill/hold dots + marginal-lift waterfall + recipe table; scrub to replay
  the ascent. Verified in-browser (no JS errors; 8.1× lift on the reco readout ladder renders).

**84 tests, 13 modules, all green.** Harness coverage now: R1, R2(spec), R3(canary), R4, R7, R8,
plus the whole BEAT_IT engine (metrics/significance/holdout/erroranalysis/recipe/contract/climb).
Remaining honest gaps for after the first real run: contamination gate for FM campaigns, R5 proxy
calibration, R6 live monitor, R2 live submit — all need real infra or a real campaign to shape.

---

## Round 15 — 2026-07-13, dollar budgets + feasibility ("is $100 even enough?")

PI asked two things: (1) will it adhere to a $100 budget, (2) will it honestly say when $100
*isn't enough* to make progress, the way labs de-risk before scaling.

**(1) Dollar budget layer** — `budget.to_a10g_hours($)` / `to_usd()` (spot $0.35 / on-demand
$1.01 per A10G-hr) turn `$100` into the A10G-equivalent hours the caps use; `preflight(...,
envelope_ah=to_a10g_hours(100))` refuses over-budget runs. **Honest caveat recorded in the
docstrings:** this is real enforcement *only* once every run goes through the launcher and
reports its actual cost to the registry, and the live `anyscale job submit` is still the stubbed
boundary — so today it's "the harness refuses + hard-kills overruns," not a physics guarantee.

**(2) `feasibility.py`** — the de-risking tool, three checks grounded in how labs actually decide:
- COMPUTE affordability: $ → A10G-hr vs the plan (gate + proxy + ≥1 full run).
- STATISTICAL power: `min_detectable_effect` vs the target gain — if you can't detect it, $100
  buys a run you can't read.
- LEARNING CURVE: `fit_power_law` on 3 cheap pilot points (error = c + a·N^-b, the scaling-laws
  form) → extrapolate data/compute to a target, and — the gem — flag when the target is **below
  the fitted floor c** (no budget reaches it; change the method, not the money).
Verdict: NOT_ENOUGH / PILOT_ONLY / GO. Demo'd: $100 on ESM-2 (wave1) → GO; on 7B-RL → PILOT_ONLY
(reproduce + proxy to de-risk, but a full run is ~$231); underpowered eval → NOT_ENOUGH; target
below the learning-curve floor → NOT_ENOUGH. **93 tests, 15 modules, all green.**

A test expectation was wrong and the code was more honest than the test (7B-RL on $100 is
PILOT_ONLY, not a flat NOT_ENOUGH — you *can* still reproduce + de-risk); fixed the test.

---

## Round 16 — 2026-07-13, de-positioning audit (per PI + research_director.md)

PI called it: sales/Anyscale-business positioning had seeped into what's meant to be a lean,
**IC-MLE-facing** research tool whose *sole* purpose is to empirically show the autoresearch loop
improves a model. Anyscale/Ray belongs only as the **compute substrate** (Ray Data / Train / Tune
ASHA / Serve — what makes the experiment matrix affordable and observable), per
`research_director.md`. Audited and stripped it everywhere.

**Removed / reframed:** the `## 2. Vertical & why Anyscale` section in `_TEMPLATE` + all 11
campaigns → `## 2. Why this is a good autoresearch testbed` (what makes a clean baseline/gate +
which Ray primitives the loop exercises). Cut every customer proof-point (Pinterest/Coinbase/
Canva/Mirakl/Attentive/Instacart), "most-winnable workload," "platform story," "playbook,"
"vertical coverage," "which campaigns ship first." Rewrote `README.md` (IC audience; single
outcome = empirical model improvement or honest finding; Ray-as-substrate; the phased loop +
gates from research_director.md). Rewrote `SEED_INDEX.md` (Domain column not Vertical; workload-
shape coverage not vertical coverage). Neutralized campaign 10 (removed PathAI-as-customer /
meeting / commercial-engagement framing → a plain digital-pathology testbed). Fixed the
`commercial_use` gate rationale (REQUIREMENTS #7, _TEMPLATE, 03, 06): the gate is about *legal use
of the output*, not "producing Anyscale marketing."

**Interpretation of "remove any kinda of rules":** I read this as the *positioning cruft*, plus
softening preachy product framing ("the program's brand," "product," "ships a blog post"). I
**kept the rigor** — reproduce-before-claim, CIs/noise floor, provenance, cost caps, the gates —
because those ARE the substance research_director.md is built on (its whole §6 is the gate table).
If "rules" was meant to include the rigor gates, say so and I'll reconsider; I judged that removing
them would gut the thing the PI explicitly called "a rigorous setup."

Verified: positioning language gone from all core docs (grep clean); 93 harness tests still green.

---

**Lit review (added to `BEAT_IT.md` → Literature):** mapped each component to real arXiv work —
AI-Scientist/Agent-Laboratory/LLM-Speedrunning (the agent space; their lesson: agents overfit
the eval), Accounting-for-Variance (2103.03098) + the Paired-Bootstrap-for-small-gains protocol
(2511.19794, ≈ our `paired_bootstrap`), Dwork Thresholdout + Blum-Hardt Ladder (≈ our holdout
budget), SliceLine/DivExplorer/DEIM (slice discovery — an upgrade path to *auto-find* the weak
subgroup), Successive-Halving/Hyperband/ASHA (our lift-per-GPU-hour allocation). Named the
remaining rigor gaps the literature flags: no program-level multiple-comparisons/FDR control,
no up-front power analysis (positives_warning is a stub), no contamination gate for FM campaigns.
