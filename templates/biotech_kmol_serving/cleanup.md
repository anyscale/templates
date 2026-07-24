# Cleanup & Pre-Share Plan — biotech_kmol_serving

**Internal only — do NOT ship to the customer.** (Lives at the template root, outside `port/`.)

**Status: planning.** Execute the fixes *after* the molecule-diversity + true-throughput
rework lands (§Molecules, §Throughput) — no point re-wording docs around numbers we're
about to replace.

## The core correction driving this plan
None of our current "throughput" figures are an **open-loop served throughput**. They are:
a forward-only compute ceiling (60k mol/s), a compute-pipeline rate (placement-group
scaling), or a **closed-loop** asyncio client (the serve numbers = "how fast one client
script drives it," not sustained requests/sec under real concurrency). Re-measure with
locust before quoting any "throughput" to Takeda.

---

## P0 — blockers (broken / unsafe / leaks internal info)
- [ ] **Scrub Anyscale infra IDs** — workspace `expwrk_…`, project `prj_…`, cloud `cld_…` appear in `port/README.md` and `port/REPRODUCE.md` (would ship to Takeda). Replace with `<your-workspace-id>` placeholders or delete the provenance lines. Keep `plan.md` / `cleanup.md` internal.
- [ ] **Restore query auth** — `port/service.yaml` and `port/service.image.yaml` dropped `query_auth_token_enabled: true` (the pre-port config had it). As-is, following the instructions deploys a **public, unauthenticated GPU endpoint**.
- [ ] **Make the deploy actually run** — `port/config.json` and `port/checkpoints/` don't exist (gitignored) but `service.yaml` + `Dockerfile` reference them → deploy/build fails as committed. Provide/point at a real `config.json` and document "stage real checkpoints here."
- [ ] **Guard synthetic-vs-real weights** — `port/kmolport/abstract_network.py` loads `strict=False` (warn-only). A key mismatch on real checkpoints would silently leave the model half-random. Make missing/unexpected keys a hard error (or `strict=True`) at production load; log a checkpoint hash + provenance at startup.

## P1 — credibility (before numbers reach the AE)
- [ ] **Molecule diversity** — replace the 10-SMILES modulo pool with the ~10k real diverse library (§Molecules). Use it in every benchmark, the parity check, AND locust.
- [ ] **True throughput** — re-measure served throughput with open-loop locust (§Throughput). Report served mol/s + p50/p99 + error rate + measured GPU util; keep it clearly separate from compute-capacity and the forward-only microbench.
- [ ] **Reconcile "one L4 = 23×"** — that run used 12 featurizer actors borrowing the head node's CPUs; a scaled-out g6.2xlarge has 8 vCPUs and runs 6 featurizers → honest per-node number is **13×**. Fix the docs.
- [ ] **Demote 60k / 353× forward-only** out of the headline; caption it "GPU headroom microbench," not throughput.
- [ ] **Fix the 170/s baseline** — undocumented and it mostly measures kMoL's per-call reload overhead, not GPU speed. Re-measure apples-to-apples (kMoL offline predict, load-once, batched, same L4) and recompute the multipliers.
- [ ] **Fix the synthetic caveat placement** — move it directly under the brief Summary; rename "Correctness / same model" → "Port fidelity (code equivalence, synthetic weights)" so it can't read as validated accuracy.

## P2 — completeness (strengthens the pitch)
- [ ] **Validate with REAL trained checkpoints** — re-run parity + a throughput sample once Takeda's weights exist. The #1 scientific gap (everything so far is synthetic).
- [ ] **Cost** — $/million molecules + L4-hour basis for the served and scaling numbers.
- [ ] **Sustained runs** — minutes, not the current ~5s bursts, to back "near-linear / sustained."
- [ ] **Atom-type match** — confirm `kmolport` `DEFAULT_ATOM_TYPES` matches Takeda's training atom set; extend parity to large / unusual-atom molecules.
- [ ] **Ship only `port/`** (scrubbed). Park the contradictory pre-port root stack (`src/`, root `service.yaml` that asserts kMoL "does NOT run on a stock image" — now disproven) out of the customer bundle.
- [ ] Save the CPU parity (0.0) to a JSON artifact (currently narrative-only).

---

## §Molecules (diversity rework)
- Build **~10k unique, real, size-diverse** SMILES (tox21 for domain relevance + ZINC/ChEMBL sample for the large-molecule tail), canonicalized + deduped + RDKit-sanitized. ~1k is the minimum; ~10k is solid; beyond that adds download weight without changing the per-molecule physics.
- **Characterize the distribution**: heavy-atom count, MW, #bonds, #rings, #aromatic atoms, SMILES length → histograms + summary stats (median / p5 / p95 / max).
- **State throughput against size**: report a size→featurization-time curve and mol/s per size bucket, so we can say "at your library's median size, expect ~Y mol/s" instead of one context-free number.

## §Throughput (true measurement rework)
Report **three clearly-labeled numbers, never conflated:**
1. **Served throughput (the headline)** — open-loop locust over HTTP against the deployed Serve app: realistic request mix, ramped load, sustained ≥ minutes. Report mol/s + p50/p99 + error rate + measured GPU util, at 1→4 GPU autoscale. *(This is REC-4 / plan P2 — written but never actually run.)*
2. **Compute-pipeline capacity** — the placement-group featurizer→GPU scaling (processing ceiling). Label it capacity, not served throughput.
3. **Forward-only GPU ceiling** — the 60k microbench; caption as headroom.

**Locust design (extend `scripts/locustfile.py`, don't rewrite):**
- Draw molecules from the **diverse ~10k library**, not the 12 hardcoded SMILES.
- Model **both request shapes**: interactive (1 molecule / request) and **screening/bulk** (N molecules / request) — the bulk case is where batching and real throughput live.
- Run **open-loop from a separate process/node**; ramp users to find the sustained max (throughput plateaus while latency climbs).
- Capture failures/timeouts; watch `nvidia-smi` concurrently to confirm featurization-bound vs GPU-bound at peak.
- (Ref reviewed: `archive/reference/triton_services/content/locustfile.py` — too minimal to borrow; our `scripts/locustfile.py` is the better base.)
