"""The three compounds named on the 2026-07-23 call as routine sanity-check inputs.

Single source of truth. Every driver that can be pointed at this set reads it from here,
so all rows of the 3-molecule comparison are measured on identical structures. Heavy-atom
counts are *asserted* at run time by `bench_three_molecules.py`, not trusted from here.

Drivers accept this set via `KMOL_POOL3=1` (`serve_bulk.py`, `gpu_run.py`) or
`--pool3` (`serve_pipeline_bulk.py`). Note these three average 29.7 heavy atoms against
the 15,751-molecule library's 41, implying ~213 mol/s per core versus the library's 131 —
so throughput measured on them reads optimistic. Don't compare it to a full-library row.
"""

TAKEDA_THREE = [
    # Minoxidil is the pyrimidine 3-N-oxide. Without the [n+]([O-]) this is a different
    # compound and comes out at 14 heavy atoms instead of 15.
    ("minoxidil", "Nc1cc(N2CCCCC2)[n+]([O-])c(N)n1"),
    ("sildenafil (Viagra)",
     "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"),
    ("atorvastatin (Lipitor)",
     "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
]

# From the brief's measured table; asserted against RDKit at run time.
EXPECTED_HEAVY_ATOMS = {
    "minoxidil": 15,
    "sildenafil (Viagra)": 33,
    "atorvastatin (Lipitor)": 41,
}

SMILES = [s for _, s in TAKEDA_THREE]
