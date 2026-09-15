# ── stage 3 : stream the batches into weighted train/test chunks ─────────────
# One rule is generated per dataset because the output paths include the
# dataset-specific short_name (e.g. z_train_00000.parquet, qq_test_00003.parquet).
# Snakemake output patterns must be statically derivable from wildcards alone,
# so a single parameterised rule cannot encode a config-lookup in its output
# path.  Generating one rule per dataset at DAG-construction time is the
# idiomatic Snakemake solution for this pattern.
#
# Row groups are pulled from the batches in turn and appended to the train or
# test fill buffer, each of which writes an output file once it holds
# chunk_size events; an input running out mid-file just means the next one is
# opened and keeps filling.
#
# How many chunk files a split needs depends on its event count, which is only
# known once the merge has run, so each split is represented by a marker file
# plus its first chunk (which always exists) rather than by every chunk.
for _ds, _cfg in DATASETS.items():
    _short = SHORT_NAMES[_ds]
    rule:
        name: f"merge_and_split_{_ds}"
        localrule: True
        input:
            # Capture _ds in the default arg to avoid the Python late-binding
            # closure problem inside a for-loop.
            batches  = lambda wc, ds=_ds: ntupelized_files_for(ds),
            sig_w    = f"{WEIGHTS_DIR}/sig_weights.npy",
            bkg_w    = f"{WEIGHTS_DIR}/bkg_weights.npy",
            p_edges  = f"{WEIGHTS_DIR}/p_edges.npy",
            th_edges = f"{WEIGHTS_DIR}/theta_edges.npy",
        output:
            # One marker per split, plus the first chunk of each, which the
            # validation stage reads directly.
            [touch(chunks_marker(_short, split)) for split in SPLITS],
            [first_chunk(_short, split) for split in SPLITS],
        params:
            input_dir   = f"{TEMP_DIR}/{_ds}",
            output_dir  = OUTPUT_DIR,
            short_name  = _short,
            train_frac  = _cfg.get("train_frac", 0.8),
            chunk_size  = CHUNK_SIZE,
            row_group   = ROW_GROUP_SIZE,
            split_seed  = SPLIT_SEED,
            side        = "sig" if _cfg["is_signal"] else "bkg",
            weights_dir = WEIGHTS_DIR,
            container   = CONTAINER,
        resources:
            mem_mb  = 32_000,
            runtime = 120,
        shell:
            """
            mkdir -p {params.output_dir}
            {params.container} python ntupelizer/scripts/merge_files.py \
                -i {params.input_dir} \
                -o {params.output_dir} \
                -s {params.short_name} \
                -f {params.train_frac} \
                -w {params.weights_dir} \
                --side {params.side} \
                --chunk-size {params.chunk_size} \
                --row-group-size {params.row_group} \
                --seed {params.split_seed}
            """
