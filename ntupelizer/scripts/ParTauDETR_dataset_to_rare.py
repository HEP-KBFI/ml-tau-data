import glob
import os

import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq

DETR_dataset_dir = "/scratch/persistent/laurits/ml-tau/20260818_tauDaughterDataset/"
rare_decays_dir = "/scratch/persistent/laurits/ml-tau/20260824_rareDecaysDataset/"

CHUNK_SIZE = 100_000  # number of rows per chunk


def get_decay_mode_id(daughter_pdgs):
    # keys = np.unique(ak.flatten(abs(arr.gen_jet_tau_vis_daughter_pdgs)))
    keys = [
        22,
        111,
        130,
        211,
        221,
        223,
        310,
        311,
        321,
        323,
    ]  # Should get the same result as above, but this is just a failsafe.

    targets = ak.Array(
        [
            # 22 111 130 211 221 223 310 311 321 323
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 0: pi + pi0
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 1: pi
            [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],  # 2: 3pi
            [0, 2, 0, 1, 0, 0, 0, 0, 0, 0],  # 3: pi + 2pi0
            [0, 1, 0, 3, 0, 0, 0, 0, 0, 0],  # 4: 3pi + pi0
            [0, 3, 0, 1, 0, 0, 0, 0, 0, 0],  # 5: pi + 3pi0
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # 6: pi + K0
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7: K
            [0, 2, 0, 3, 0, 0, 0, 0, 0, 0],  # 8: 3pi + 2pi0
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # 9: K + pi0
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # 10: pi + pi0 + K0
            [0, 0, 0, 2, 0, 0, 0, 0, 1, 0],  # 11: 2pi + K
        ]
    )

    counts = ak.zip({f"n_{k}": ak.sum(abs(daughter_pdgs) == k, axis=1) for k in keys})
    signature = ak.zeros_like(counts.n_22)
    for k in keys:
        signature = signature * 10 + counts[f"n_{k}"]

    # Encode the target configurations using the exact same scheme.
    target_signature = ak.zeros_like(targets[:, 0])

    for i in range(len(keys)):
        target_signature = target_signature * 10 + targets[:, i]

    # Match each jet against the 12 target signatures.
    matches = signature[:, None] == target_signature[None, :]

    # ID of matching target.
    class_id = ak.argmax(matches, axis=1, mask_identity=False)

    # No match -> Other = -1
    class_id = ak.where(
        ak.any(matches, axis=1),
        class_id,
        15,
    )
    return class_id


os.makedirs(rare_decays_dir, exist_ok=True)

for path in sorted(glob.glob(os.path.join(DETR_dataset_dir, "*.parquet"))):
    basename = os.path.basename(path)
    if "qq" in basename:
        continue
    stem = os.path.splitext(basename)[0]
    is_qq = "qq" in basename
    print(f"Processing {basename}")
    pf = pq.ParquetFile(path)
    for i, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE)):
        data = ak.from_arrow(pa.Table.from_batches([batch]))

        # Temporarily added to clean: drop jets with an electron daughter.
        mask = ak.sum(abs(data.gen_jet_tau_vis_daughter_pdgs) == 11, axis=1) == 0
        data = data[mask]

        if is_qq:
            class_id = ak.Array([-1] * len(data))
        else:
            class_id = get_decay_mode_id(data.gen_jet_tau_vis_daughter_pdgs)

        data = ak.with_field(data, class_id, "gen_jet_tau_decay_mode_rare")

        if len(data) == 0:
            continue

        out_path = os.path.join(rare_decays_dir, f"{stem}_{i:05d}.parquet")
        ak.to_parquet(data, out_path)
        print(f"  -> {out_path}")
