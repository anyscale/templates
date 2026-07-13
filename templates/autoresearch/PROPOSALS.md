# PROPOSALS — reviewer's proposed follow-ups

Written by the second (reviewer) Claude session after the first author session went quiet
following commit `56d9896c`. These are **proposals for the PI/author to accept or reject**,
not edits applied to the authored docs — keeping the reviewer's hands off the author's
files on purpose (see `critiques.md` #22 on why reviewer/author independence matters).

Of the four "promote a good instinct to an enforced rule" critiques, the author already
folded in most (#1 n=1 guard, #3 typed decisions, #4 single-writer, #5 proxy arithmetic,
#6 repo intent, #8 pre-registration freeze, #21 mechanism check). Three remain. One is now
settled in code; two need a PI call.

---

## P1 — R1 registry reference implementation (DONE, in `harness/`)

`harness/registry.py` + `harness/test_registry.py` — the "couple hundred lines of Python"
the build-order rule says to write first. Dependency-free stdlib. `python3 test_registry.py`
→ 11 passing tests, each one an executable REQUIREMENTS-R1 clause. It implements the schema
verbatim and adds exactly one thing beyond the current docs: **tier-weighted cost** (P2).

This is a *reference* implementation to react to, not a merge request — the author owns
whether it becomes the real R1.

## P2 — Tier-weight the budget currency (critiques.md #14/#20) — now demonstrated in code

BUDGET_POLICY states envelopes/caps in raw GPU-hours, but an A100-hour costs 3.5 A10G-hours
and an H100-hour 5.5. `registry.py` stores `a10g_equiv_hours` alongside raw hours, and the
test `test_the_7b_rl_run_is_miswaved` proves the consequence: campaign 02's "~120 H100-hr"
full run is **660 A10G-equivalent hours → Wave 3**, not the Wave 2 its plan claims (which
would trigger a lighter approval ritual on the program's single most expensive run).

**Proposed one-line BUDGET_POLICY edit** (under "The three rungs"):
> Envelopes and rung caps are denominated in **A10G-equivalent GPU-hours** — raw hours ×
> the rel-cost column of the conversion table (T4 ×0.5, L4 ×0.8, A10G ×1.0, L40S ×1.8,
> A100 ×3.5, H100 ×5.5). The registry stores both; R4 enforces against the A10G-equivalent.

Then re-derive the wave labels: 02 and 05/06 stop being vibes and become arithmetic.

## P3 — A commercial-use license gate (critiques.md #19) — needs a PI decision

The program exists to make Anyscale marketing, yet "may we legally market this output?" is
still per-campaign memory. The author hand-caught four landmines (Criteo CC-BY-NC blocks
marketing; OpenVLA weights under Llama-2's >700M-MAU clause; ESM-3 non-commercial; MovieLens
research-only). That is exactly the informal human-in-the-loop the program is meant to
mechanize.

**Proposed:** add a required field to `_TEMPLATE.md` §3 and make it a pre-registration gate
in REQUIREMENTS alongside the reproduction gate:
> **`commercial_use:` `yes` | `no` | `needs-legal`** — can the reference's code AND weights
> AND dataset licenses support the campaign's stated (marketing) purpose? A `no`/`needs-legal`
> blocks budget until the PI signs the license risk. No spending on an unshippable result.

## P4 — Define "cost/efficiency beat" as an honest claim type (critiques.md #18) — needs a PI call

Campaigns 04/07/09 headline a "beat on cost/throughput," but the references published no
cost number — so the baseline is self-built and can be made as slow as you like.

**Proposed** (one place — REQUIREMENTS non-functional gates):
> A **cost/efficiency beat** is a distinct claim type from a metric beat. Its baseline must
> be *a competent single-GPU implementation given the same hardware budget, reasonably
> tuned* — never the reference's demo script — and the quality metric (nDCG/WQL/WER) must be
> held equal within CI. Blog it as an efficiency result, not as beating the published number.

---

## Not addressed here (deliberately)

Campaign-specific critiques (#9–13, #16, #17 — BEIR strawman framing, E5 pooling mechanism,
proxy-axis split, MSMARCO RAM budget, OpenVLA sim-fleet decoupling) live in `critiques.md`
per campaign. They're for whoever runs that campaign, not program-wide policy.
