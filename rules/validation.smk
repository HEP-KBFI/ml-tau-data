# ── stage 4 : validation ──────────────────────────────────────────────────────
# Single global job comparing the signal and background distributions, including
# the weight distributions that apply_weights.py -p used to produce.
rule validation:
    input:
        # Plots are made from the first train chunk of each dataset: the
        # validate_ntuples.py plots load their input in full, and one chunk
        # (CHUNK_SIZE jets) is plenty of statistics for distribution shapes.
        chunks = [first_chunk(SHORT_NAMES[ds], "train") for ds in DATASETS],
        p_edges = f"{WEIGHTS_DIR}/p_edges.npy",
        theta_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
        script = "ntupelizer/scripts/validate_ntuples.py",
    output:
        touch(f"{OUTPUT_DIR}/validation/.done")
    params:
        outdir    = f"{OUTPUT_DIR}/validation",
        sig_file  = first_chunk(SHORT_NAMES[SIG_DATASET], "train"),
        bkg_args  = " ".join(f"-b {SHORT_NAMES[ds]}:{first_chunk(SHORT_NAMES[ds], 'train')}" for ds in BKG_DATASETS),
        weights_dir = WEIGHTS_DIR,
        container = CONTAINER,
    resources:
        mem_mb  = 32_000,
        runtime = 60,
    shell:
        """
        mkdir -p {params.outdir}
        {params.container} python ntupelizer/scripts/validate_ntuples.py \
            -s {params.sig_file} \
            {params.bkg_args} \
            -w {params.weights_dir} \
            -o {params.outdir}
        """
