# ── stage 4 : validation ──────────────────────────────────────────────────────
# Single global job comparing the signal and background distributions, including
# the weight distributions that apply_weights.py -p used to produce.
rule validation:
    input:
        # Plots are made from the first train chunk of each dataset: the
        # validate_ntuples.py plots load their input in full, and one chunk
        # (CHUNK_SIZE jets) is plenty of statistics for distribution shapes.
        [first_chunk(SHORT_NAMES[ds], "train") for ds in DATASETS]
    output:
        touch(f"{OUTPUT_DIR}/validation/.done")
    params:
        outdir    = f"{OUTPUT_DIR}/validation",
        sig_file  = first_chunk(SHORT_NAMES[SIG_DATASET], "train"),
        bkg_file  = first_chunk(SHORT_NAMES[BKG_DATASET], "train"),
        container = CONTAINER,
    resources:
        mem_mb  = 32_000,
        runtime = 60,
    shell:
        """
        mkdir -p {params.outdir}
        {params.container} python ntupelizer/scripts/validate_ntuples.py \
            -s {params.sig_file} \
            -b {params.bkg_file} \
            -o {params.outdir}
        """
