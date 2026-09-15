# ── stage 2 : compute weight matrices (single global job) ────────────────────
# Weights are computed by comparing the signal and background (p, theta)
# distributions.  Only the gen_jet_p4 column is read, so this runs directly on
# the ntupelized batches — before any merging — which is what allows stage 3 to
# apply the weights in the same pass that writes its output.
rule compute_weights:
    input:
        sig = lambda wc: ntupelized_files_for(SIG_DATASET),
        bkg = lambda wc: ntupelized_files_for(BKG_DATASET),
    output:
        sig_w    = f"{WEIGHTS_DIR}/sig_weights.npy",
        bkg_w    = f"{WEIGHTS_DIR}/bkg_weights.npy",
        p_edges  = f"{WEIGHTS_DIR}/p_edges.npy",
        th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
    params:
        output_dir     = WEIGHTS_DIR,
        sig_dir        = f"{TEMP_DIR}/{SIG_DATASET}",
        bkg_dir        = f"{TEMP_DIR}/{BKG_DATASET}",
        produce_plots  = config.get("weights", {}).get("produce_plots", False),
        n_files        = config.get("weights", {}).get("n_files_per_sample", -1),
        container      = CONTAINER,
    resources:
        mem_mb  = 32_000,
        runtime = 30,
    shell:
        """
        mkdir -p {params.output_dir}
        {params.container} python ntupelizer/scripts/compute_weights.py \
            -i {params.sig_dir} \
            -b {params.bkg_dir} \
            -o {params.output_dir} \
            -n {params.n_files} \
            $([ '{params.produce_plots}' = 'True' ] && echo '-p' || true)
        """
