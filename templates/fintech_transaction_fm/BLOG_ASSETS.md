# BLOG_DRAFT.md — asset checklist (figures, stats, sign-offs)

Every open `[B#]`/`[TODO]` in the draft, with its data source and what's
needed to close it. Tags: **[local]** = buildable now from numbers already
in the draft/artifact JSONs; **[cluster]** = needs the fintech_fm workspace
(`$BASE` = /mnt/user_storage/transaction-fm-v2; `anyscale ssh --id ...`, see
memory note on cloud scoping); **[person]** = needs a human.
Reuse column: does the asset carry into the follow-up "why Anyscale wins"
post? (P = personal-blog only, A = Anyscale post too.)

## Figures (6 required, 1 optional)

- [ ] **B1 — hero: two-panel dot + 95% CI** [local] [reuse: A]
  Left panel: their-protocol 100k — every table-1 row as dot+CI whiskers,
  horizontal reference lines at NVIDIA published baseline 0.1238 and fusion
  0.1755. Right panel: full test period — baseline 0.2081 vs embed_xgb
  0.2788 / 0.3027 / 0.2665 (512/1024/2048), CI whiskers. Shared y-axis
  (AP), caption: "different eval sets — never compare across panels."
  Data: all numbers in draft tables 1+2; canonical sources
  `$BASE/downstream/*/bootstrap_ci.json` + `*_fulltest` dirs.

- [ ] **B2 — pipeline architecture graphic** [local] [reuse: A — the anchor
  figure of the Anyscale post]
  Redraw the README ASCII diagram (raw parquet → Ray Data vocab → Ray Data
  tokenize → Ray Train DDP pretrain → Ray Data GPU embed → XGBoost fraud +
  reco rank → Ray Serve). For the Anyscale post, annotate each stage with
  its scale knob + hardware.

- [ ] **B7 — table-2 dot-and-CI plot** [local] [reuse: P]
  All 8 table-2 rows, baseline as reference band. Data: in draft; verify
  against `*_fulltest` metrics JSONs before export.

- [ ] **B9 — burst histogram** [cluster] [reuse: P]
  Histogram of distance (in same-card transactions) to previous fraud, test
  frauds vs normals (log-x). Needs a small script over the raw parquet
  (24.4M rows — run on the workspace, not laptop). Blocked by B8 (same
  computation produces both). No existing artifact — this is the one
  figure whose *data* doesn't exist yet.

- [ ] **B16 — month-canary training curves** [cluster] [reuse: P]
  `field_ce/month` (and/or per-field macro acc) across the 512/1024/2048
  runs on one axis — shows the 512 phase transition the longer runs never
  reach; optionally the 2048 20→40-epoch continuation grinding. Source:
  TensorBoard event files `$BASE/tensorboard/…` — export scalars to CSV,
  plot locally. Draft cites macro 0.824/0.770/0.754 @20ep, 2048@40 → 0.767.
  (B16-alt slot in the Cost section: use leftover TB curves or drop.)

- [ ] **B-RECO-FIG — reco slices bar chart** [local] [reuse: P]
  Grouped bars, HR@10: overall (naive 0.6474 vs hybrid 0.6582),
  next∉top-10 = 35.3% of events (naive 0.000 vs model 0.158), never-seen
  6.3% (history methods 0 vs full-vocab MLP 0.077). Source:
  `$BASE/downstream/full_nextmerchant/probe_metrics_v3.json`.

- [ ] **B11 (OPTIONAL) — surprise-vector diagnostic anecdote** [cluster]
  Decide cut vs keep before spending effort; the section reads fine
  without it. If kept: needs artifacts from the research-branch probe era.

## Stats / numbers to fetch or verify

- [ ] **B8 — ledger the burst stat** [cluster]
  "90% of test frauds have a prior same-card fraud within the preceding
  512 transactions vs 7.3% of normals" is currently from campaign memory,
  not a checked-in artifact. Write the script (same pass as B9), save the
  JSON, cite exact numbers in the draft.

- [ ] **B10 — velocity control table** [cluster→verify only]
  Three rows: raw13 / raw13+velocity / embed_xgb. Numbers exist
  (`$BASE/downstream/full_velocity/velocity_metrics.json`; draft cites
  0.0757) — pull exact values for all three rows and inline the table.

- [ ] **B15 — exact $ per run** [cluster/console] [reuse: A — headline
  material for the Anyscale post]
  Pull actual job costs from the Anyscale console for: 512 full run, 1024
  (`prodjob_1nxzlze72xgv7imvgvlw6hk9eb`), 2048
  (`prodjob_szkm6hlic3hkhgrn4admydltua`) + 40-ep continuation, pinned+CUDA
  eval (`prodjob_n8tv97crr4nasnsjmw1ne3x2yg`), fulltest evals. Draft
  placeholder: "$15–25 per headline run."

- [ ] **Re-verify inline tables against artifacts** [cluster→verify only]
  Table 1 ↔ `bootstrap_ci.json`; Table 2 ↔ `*_fulltest`; paired ordering ↔
  `downstream/paired_bootstrap_embed_xgb.json`; reco ladder + hybrid ↔
  `probe_metrics{,_v2,_v3}.json`; shuffled control ↔
  `full_probe/probe_metrics_shuffled_seed0.json`. One pass, before figures
  are exported, so figures and text can't drift apart.

## Sign-offs / links (people + publish events)

- [ ] **Zach: repro-branch numbers + framing** [person]
  Confirms 0.0614 (his retrained fm-only) vs 0.0123 (their published
  weights), and which benchmark sample his final table used (his split vs
  our 1M-balanced read — ZGARNER_INTEGRATION open item #1).
- [ ] **Zach: single-txn encoding confirmation** [person]
  ZGARNER open item #2 (fm rows are NB04-style, max_ctx≈14). The ablation
  "4x short" comparison and the new "What we didn't run" takeaway lean on
  this.
- [ ] **Link: Zach's faithful-repro branch/appendix** (two TODOs, lines
  ~26 and ~103).
- [ ] **Link: published template** (line ~159) — blocked on template
  publish, which is blocked on the known CI failure (`prepare_tabformer`
  unique() 28GB HashAggregate; see memory). Separate track from the blog.

## Venue decisions (now that this is Geoff's personal-blog version)

- [ ] CTA block (line ~173): swap Anyscale-house CTA for personal framing
  (disclosure line: built at Anyscale; template link as the CTA).
- [ ] Voice pass: "we" → decide "I/we" mix; keep Zach credited.
- [ ] Title check: current title leads with "on Ray" — fine for personal
  post; the Anyscale-angle version gets its own title later.

## Follow-up post ("why Anyscale wins", written after this one)

Not started — separate short draft. Carries over: B2 (annotated), B15
(cost table), the scale-knob table (line ~157 region of draft), the
"laptop→cluster is a config change" thread, tokenization-scales-horizontally
vs single-GPU-RAPIDS contrast, and the 0→8 autoscaled fulltest eval story.
Cut: all mechanism/benchmark-honesty sections (link to the personal post
instead).
