"""One-time preprocessing script: converts parquet files to pre-built tensor .pt files.

Run once before training:
    python ntupelizer/scripts/preprocess_torch.py -i <output_dir> [--max-cands 20]

Or via Snakemake (rule preprocess_torch).

Re-run whenever build_tensors logic changes (new features, normalization, etc.).
The output .pt files live alongside the parquet files with the same stem.
"""

import os
import glob
import argparse
import multiprocessing as mp
import time
from tqdm import tqdm
import torch
import numpy as np
import awkward as ak

from ntupelizer.tools import general as g
from ntupelizer.tools import features as f
from ntupelizer.tools.tau_decaymode import get_reduced_decaymodes

# Reduced decay mode → class index (0-5) for 6-class one-hot encoding
_DM_TO_CLASS = {0: 0, 1: 1, 2: 2, 10: 3, 11: 4, 15: 5, 16: 5}


def prepare_one_hot_encoding(reduced_decaymodes: np.ndarray) -> np.ndarray:
    return np.vectorize(_DM_TO_CLASS.get)(reduced_decaymodes)


def stack_and_pad_features(features: ak.Array, max_cands: int) -> np.ndarray:
    padded = [
        ak.to_numpy(
            ak.fill_none(ak.pad_none(features[field], max_cands, clip=True), 0.0)
        )
        for field in features.fields
    ]
    return np.stack(padded, axis=-1)


def build_tensors_from_data(data: ak.Array, max_cands: int) -> tuple:
    """Build all training tensors from one row group of raw parquet data.
    Returns a tuple of 8 items — all torch.Tensors or dicts of torch.Tensors.
    This is the single source of truth used by both the live dataloader and
    the preprocessing script."""
    jet_constituent_p4s = g.reinitialize_p4(data.reco_cand_p4s)
    gen_jet_tau_p4s = g.reinitialize_p4(data.gen_jet_tau_p4)
    jet_p4s = g.reinitialize_p4(data.reco_jet_p4)
    gen_jet_p4s = g.reinitialize_p4(data.gen_jet_p4)

    eps = 1e-6
    cand_features = ak.Array(
        {
            "cand_deta": f.signedDeltaEta(jet_constituent_p4s.eta, jet_p4s.eta),
            "cand_dphi": f.signedDeltaPhi(jet_constituent_p4s.phi, jet_p4s.phi),
            "cand_logpt": np.log(np.maximum(jet_constituent_p4s.pt, eps)),
            "cand_loge": np.log(np.maximum(jet_constituent_p4s.energy, eps)),
            "cand_logptrel": np.log(
                np.maximum(jet_constituent_p4s.pt / jet_p4s.pt, eps)
            ),
            "cand_logerel": np.log(
                np.maximum(jet_constituent_p4s.energy / jet_p4s.energy, eps)
            ),
            "cand_deltaR": f.deltaR_etaPhi(
                jet_constituent_p4s.eta,
                jet_constituent_p4s.phi,
                jet_p4s.eta,
                jet_p4s.phi,
            ),
            "cand_charge": data.reco_cand_charges,
            "isElectron": ak.values_astype(abs(data.reco_cand_pdgs) == 11, np.float32),
            "isMuon": ak.values_astype(abs(data.reco_cand_pdgs) == 13, np.float32),
            "isPhoton": ak.values_astype(abs(data.reco_cand_pdgs) == 22, np.float32),
            "isChargedHadron": ak.values_astype(
                abs(data.reco_cand_pdgs) == 211, np.float32
            ),
            "isNeutralHadron": ak.values_astype(
                abs(data.reco_cand_pdgs) == 130, np.float32
            ),
            "cand_dz": data.reco_cand_dz,
            "cand_dz_error": data.reco_cand_dz / data.reco_cand_dz_error,
            "cand_dxy": data.reco_cand_dxy,
            "cand_dxy_error": data.reco_cand_dxy / data.reco_cand_dxy_error,
        }
    )

    cand_kinematics = ak.Array(
        {
            "cand_px": jet_constituent_p4s.px,
            "cand_py": jet_constituent_p4s.py,
            "cand_pz": jet_constituent_p4s.pz,
            "cand_en": jet_constituent_p4s.energy,
        }
    )

    if "cls_weight" not in data.fields:
        weight_tensors = torch.ones(len(data), dtype=torch.float32)
    else:
        weight_tensors = torch.from_numpy(
            ak.to_numpy(data.cls_weight).astype(np.float32)
        )

    gen_jet_tau_decaymode = ak.to_numpy(data.gen_jet_tau_decaymode)
    reduced_gen_decay_modes = get_reduced_decaymodes(gen_jet_tau_decaymode)
    ohe_prepared_decay_modes = prepare_one_hot_encoding(reduced_gen_decay_modes)
    gen_jet_tau_decaymode_reduced = torch.from_numpy(
        ohe_prepared_decay_modes.astype(np.int64)
    )
    gen_jet_tau_decaymode_ohe = torch.nn.functional.one_hot(
        gen_jet_tau_decaymode_reduced, 6
    ).float()
    gen_jet_tau_decaymode_exists = torch.from_numpy(
        (gen_jet_tau_decaymode != -1).astype(np.int64)
    )
    charge_tensor = torch.from_numpy(
        (ak.to_numpy(data.gen_jet_tau_charge).astype(np.int32) == 1).astype(np.float32)
    )

    _pt_gen = ak.to_numpy(gen_jet_tau_p4s.pt).astype(np.float32)
    _pt_reco = ak.to_numpy(jet_p4s.pt).astype(np.float32)
    _eta_gen = ak.to_numpy(gen_jet_tau_p4s.eta).astype(np.float32)
    _eta_reco = ak.to_numpy(jet_p4s.eta).astype(np.float32)
    _phi_gen = ak.to_numpy(gen_jet_tau_p4s.phi).astype(np.float32)
    _phi_reco = ak.to_numpy(jet_p4s.phi).astype(np.float32)
    _energy_gen = ak.to_numpy(gen_jet_tau_p4s.energy).astype(np.float32)
    _energy_reco = ak.to_numpy(jet_p4s.energy).astype(np.float32)

    _deta = _eta_gen - _eta_reco

    # _dsinphi = np.sin(_phi_gen) - np.sin(_phi_reco)
    # _dcosphi = np.cos(_phi_gen) - np.cos(_phi_reco)

    _sin_dphi = np.sin(_phi_gen - _phi_reco)
    _cos_dphi = np.cos(_phi_gen - _phi_reco)

    _vis_pt_ratio = np.maximum(_pt_gen / np.maximum(_pt_reco, eps), eps)
    _mass_gen = np.sqrt(
        np.maximum(_energy_gen**2 - (_pt_gen * np.cosh(_eta_gen)) ** 2, 0.0)
    )
    _mass_reco = np.sqrt(
        np.maximum(_energy_reco**2 - (_pt_reco * np.cosh(_eta_reco)) ** 2, 0.0)
    )
    _vis_m_ratio = np.maximum(_mass_gen / np.maximum(_mass_reco, eps), eps)
    kinematics_tensor = torch.from_numpy(
        np.stack(
            [
                np.log(_vis_pt_ratio),
                _deta,
                # _dsinphi,
                # _dcosphi,
                _sin_dphi,
                _cos_dphi,
                np.log(_vis_m_ratio),
            ],
            axis=-1,
        )
    )

    cand_kinematics_tensor = torch.from_numpy(
        stack_and_pad_features(cand_kinematics, max_cands).astype(np.float32)
    )
    cand_features_tensor = torch.from_numpy(
        stack_and_pad_features(cand_features, max_cands).astype(np.float32)
    )

    mask = (
        torch.from_numpy(
            ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(
                        ak.ones_like(data.reco_cand_pdgs), max_cands, clip=True
                    ),
                    0,
                )
            )
        )
        .to(torch.bool)
        .unsqueeze(1)
    )

    def _p4_to_tensor_dict(p4) -> dict:
        """Convert a vector.awk p4 array to a dict of float32 tensors."""
        return {
            "pt": torch.from_numpy(ak.to_numpy(p4.pt).astype(np.float32)),
            "eta": torch.from_numpy(ak.to_numpy(p4.eta).astype(np.float32)),
            "phi": torch.from_numpy(ak.to_numpy(p4.phi).astype(np.float32)),
            "energy": torch.from_numpy(ak.to_numpy(p4.energy).astype(np.float32)),
        }

    return (
        cand_features_tensor,
        cand_kinematics_tensor,
        {
            "kinematics": kinematics_tensor.float(),
            "decay_mode": gen_jet_tau_decaymode_ohe.float(),
            "charge": charge_tensor.float(),
            "is_tau": gen_jet_tau_decaymode_exists.long(),
        },
        mask,
        weight_tensors.float(),
        _p4_to_tensor_dict(gen_jet_tau_p4s),
        _p4_to_tensor_dict(jet_p4s),
        _p4_to_tensor_dict(gen_jet_p4s),
    )


def _to_numpy(item):
    """Recursively convert tensors to numpy arrays for safe inter-process pickling."""
    if isinstance(item, torch.Tensor):
        return item.numpy()
    if isinstance(item, dict):
        return {k: _to_numpy(v) for k, v in item.items()}
    if isinstance(item, tuple):
        return tuple(_to_numpy(v) for v in item)
    return item


def _to_tensor(item):
    """Recursively convert numpy arrays back to tensors."""
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item)
    if isinstance(item, dict):
        return {k: _to_tensor(v) for k, v in item.items()}
    if isinstance(item, tuple):
        return tuple(_to_tensor(v) for v in item)
    return item


def _process_row_group_batch(args: tuple) -> tuple:
    """Top-level worker: load a batch of row groups in one read and return numpy arrays.
    Returns (last_rg_idx, concatenated_numpy_tensors)."""
    parquet_path, rg_indices, max_cands = args
    data = ak.from_parquet(parquet_path, row_groups=rg_indices)
    return rg_indices[-1], _to_numpy(build_tensors_from_data(data, max_cands))


def _cat(items):
    """Concatenate a list of tensors or dicts of tensors along dim 0."""
    if isinstance(items[0], torch.Tensor):
        return torch.cat(items, dim=0)
    elif isinstance(items[0], dict):
        return {k: _cat([d[k] for d in items]) for k in items[0]}
    else:
        raise TypeError(f"Unexpected type in preprocess cat: {type(items[0])}")


def _write_progress_marker(
    progress_path: str,
    parquet_path: str,
    completed: int,
    num_row_groups: int,
    last_rg_idx: int,
    elapsed_seconds: float,
):
    """Overwrite a small marker file with the latest completed row-group status."""
    tmp_path = f"{progress_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"file={parquet_path}",
                    f"completed={completed}/{num_row_groups}",
                    f"last_row_group={last_rg_idx + 1}",
                    f"elapsed_seconds={elapsed_seconds:.1f}",
                ]
            )
            + "\n"
        )
    os.replace(tmp_path, progress_path)


def preprocess_file(
    parquet_path: str,
    max_cands: int,
    overwrite: bool = False,
    num_workers: int = 1,
    heartbeat_seconds: int = 60,
    verbose_row_groups: bool = False,
    write_progress_marker: bool = True,
    rg_batch_size: int = 32,
):
    pt_path = parquet_path.replace(".parquet", ".pt")
    if os.path.exists(pt_path) and not overwrite:
        print(f"  skip (exists): {pt_path}")
        return

    metadata = ak.metadata_from_parquet(parquet_path)
    num_row_groups = metadata["num_row_groups"]

    # Batch row groups so each worker reads N groups in a single ak.from_parquet call,
    # dramatically reducing per-task file-open / seek overhead on files with many small row groups.
    all_indices = list(range(num_row_groups))
    batches = [
        all_indices[i : i + rg_batch_size]
        for i in range(0, num_row_groups, rg_batch_size)
    ]
    num_batches = len(batches)
    args_list = [(parquet_path, batch, max_cands) for batch in batches]
    desc = os.path.basename(parquet_path)
    t0 = time.monotonic()
    progress_path = f"{pt_path}.progress"

    if write_progress_marker:
        _write_progress_marker(
            progress_path,
            parquet_path,
            completed=0,
            num_row_groups=num_row_groups,
            last_rg_idx=-1,
            elapsed_seconds=0.0,
        )
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            all_tensors = []
            per_batch_seconds = []
            completed_batches = 0
            completed_rgs = 0
            last_completed_time = t0
            iterator = pool.imap_unordered(_process_row_group_batch, args_list, chunksize=1)
            with tqdm(total=num_batches, desc=desc, unit="batch") as pbar:
                while completed_batches < num_batches:
                    try:
                        last_rg_idx, r = iterator.next(timeout=heartbeat_seconds)
                    except mp.TimeoutError:
                        now = time.monotonic()
                        stalled_for = now - last_completed_time
                        elapsed = now - t0
                        print(
                            f"  heartbeat: {completed_batches}/{num_batches} batches done "
                            f"({completed_rgs}/{num_row_groups} rgs); "
                            f"no completion for {stalled_for:.1f}s; elapsed {elapsed/60.0:.1f} min"
                        )
                        continue

                    now = time.monotonic()
                    batch_duration = now - last_completed_time
                    last_completed_time = now
                    completed_batches += 1
                    completed_rgs = min(completed_batches * rg_batch_size, num_row_groups)
                    per_batch_seconds.append(batch_duration)

                    tensor_tuple = _to_tensor(r)
                    all_tensors.append(tensor_tuple)
                    pbar.update(1)

                    n_jets_batch = tensor_tuple[0].shape[0]
                    avg_batch = float(np.mean(per_batch_seconds))
                    pbar.set_postfix_str(
                        f"rgs={completed_rgs}/{num_row_groups} jets={n_jets_batch:,} avg={avg_batch:.1f}s/batch"
                    )
                    if write_progress_marker:
                        _write_progress_marker(
                            progress_path,
                            parquet_path,
                            completed=completed_rgs,
                            num_row_groups=num_row_groups,
                            last_rg_idx=last_rg_idx,
                            elapsed_seconds=now - t0,
                        )
                    if verbose_row_groups:
                        print(
                            f"  batch {completed_batches}/{num_batches} done in {batch_duration:.1f}s "
                            f"(rg {last_rg_idx + 1}/{num_row_groups}, {n_jets_batch:,} jets, avg {avg_batch:.1f}s)"
                        )
    else:
        all_tensors = []
        per_batch_seconds = []
        completed_rgs = 0
        with tqdm(total=num_batches, desc=desc, unit="batch") as pbar:
            for batch_idx, (parquet_path_i, rg_indices, max_cands_i) in enumerate(args_list):
                batch_start = time.monotonic()
                last_rg_idx, raw = _process_row_group_batch((parquet_path_i, rg_indices, max_cands_i))
                tensor_tuple = _to_tensor(raw)
                all_tensors.append(tensor_tuple)
                pbar.update(1)

                batch_duration = time.monotonic() - batch_start
                per_batch_seconds.append(batch_duration)
                completed_rgs = min((batch_idx + 1) * rg_batch_size, num_row_groups)
                n_jets_batch = tensor_tuple[0].shape[0]
                avg_batch = float(np.mean(per_batch_seconds))
                pbar.set_postfix_str(
                    f"rgs={completed_rgs}/{num_row_groups} jets={n_jets_batch:,} avg={avg_batch:.1f}s/batch"
                )
                if write_progress_marker:
                    _write_progress_marker(
                        progress_path,
                        parquet_path,
                        completed=completed_rgs,
                        num_row_groups=num_row_groups,
                        last_rg_idx=last_rg_idx,
                        elapsed_seconds=time.monotonic() - t0,
                    )
                if verbose_row_groups:
                    print(
                        f"  batch {batch_idx + 1}/{num_batches} done in {batch_duration:.1f}s "
                        f"(rg {last_rg_idx + 1}/{num_row_groups}, {n_jets_batch:,} jets, avg {avg_batch:.1f}s)"
                    )

    merged = tuple(
        _cat([t[i] for t in all_tensors]) for i in range(len(all_tensors[0]))
    )
    torch.save(merged, pt_path)
    n_jets = merged[0].shape[0]
    total_minutes = (time.monotonic() - t0) / 60.0
    if write_progress_marker:
        _write_progress_marker(
            progress_path,
            parquet_path,
            completed=num_row_groups,
            num_row_groups=num_row_groups,
            last_rg_idx=max(num_row_groups - 1, 0),
            elapsed_seconds=time.monotonic() - t0,
        )
    print(f"  saved {n_jets:,} jets \u2192 {pt_path}")
    print(f"  total preprocess time: {total_minutes:.1f} min")


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files to pre-built .pt tensor files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Directory containing *_train.parquet and *_test.parquet files.",
    )
    parser.add_argument(
        "--max-cands",
        type=int,
        default=20,
        help="Maximum number of jet candidates to keep (default: 20).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process even if .pt already exists.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes per file (default: 1). Use os.cpu_count() for maximum parallelism.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Print heartbeat if no row group completes within this many seconds (default: 60).",
    )
    parser.add_argument(
        "--verbose-row-groups",
        action="store_true",
        help="Print one line per completed row group (default: off).",
    )
    parser.add_argument(
        "--no-progress-marker",
        action="store_true",
        help="Disable writing <output>.pt.progress marker file with latest completed row group.",
    )
    parser.add_argument(
        "--rg-batch-size",
        type=int,
        default=32,
        help="Number of row groups to read per worker task (default: 32). "
             "Higher values reduce seek overhead on files with many small row groups.",
    )
    args = parser.parse_args()

    patterns = [
        os.path.join(args.input_dir, "*_train.parquet"),
        os.path.join(args.input_dir, "*_test.parquet"),
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))

    if not all_files:
        print(f"No parquet files found under {args.input_dir}")
        return

    sorted_files = sorted(all_files)
    print(f"Found {len(sorted_files)} parquet files to process.")
    for i, path in enumerate(sorted_files, 1):
        print(f"[{i}/{len(sorted_files)}] {os.path.basename(path)}")
        preprocess_file(
            path,
            args.max_cands,
            overwrite=args.overwrite,
            num_workers=args.num_workers,
            heartbeat_seconds=max(1, args.heartbeat_seconds),
            verbose_row_groups=args.verbose_row_groups,
            write_progress_marker=not args.no_progress_marker,
            rg_batch_size=max(1, args.rg_batch_size),
        )

    print("Done.")


if __name__ == "__main__":
    main()
