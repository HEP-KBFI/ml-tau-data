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

    _dsinphi = np.sin(_phi_gen) - np.sin(_phi_reco)
    _dcosphi = np.cos(_phi_gen) - np.cos(_phi_reco)

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
                _dsinphi,
                _dcosphi,
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


def _cat(items):
    """Concatenate a list of tensors or dicts of tensors along dim 0."""
    if isinstance(items[0], torch.Tensor):
        return torch.cat(items, dim=0)
    elif isinstance(items[0], dict):
        return {k: _cat([d[k] for d in items]) for k in items[0]}
    else:
        raise TypeError(f"Unexpected type in preprocess cat: {type(items[0])}")


def preprocess_file(parquet_path: str, max_cands: int, overwrite: bool = False):
    pt_path = parquet_path.replace(".parquet", ".pt")
    if os.path.exists(pt_path) and not overwrite:
        print(f"  skip (exists): {pt_path}")
        return

    metadata = ak.metadata_from_parquet(parquet_path)
    num_row_groups = metadata["num_row_groups"]

    all_tensors = []
    for rg_idx in range(num_row_groups):
        data = ak.from_parquet(parquet_path, row_groups=[rg_idx])
        tensors = build_tensors_from_data(data, max_cands)
        all_tensors.append(tensors)

    merged = tuple(
        _cat([t[i] for t in all_tensors]) for i in range(len(all_tensors[0]))
    )
    torch.save(merged, pt_path)
    n_jets = merged[0].shape[0]
    print(f"  saved {n_jets:,} jets \u2192 {pt_path}")


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

    print(f"Found {len(all_files)} parquet files to process.")
    for i, path in enumerate(sorted(all_files), 1):
        print(f"[{i}/{len(all_files)}] {os.path.basename(path)}")
        preprocess_file(path, args.max_cands, overwrite=args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()
