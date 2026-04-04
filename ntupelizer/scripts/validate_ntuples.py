"""Produce validation plots for the weighted ntuple outputs.

Reads the flat weighted parquet files produced by apply_weights.py, then:
  - calls plot_weighting_results  : pT / eta / theta / p distributions,
                                    unweighted and weighted, signal vs bkg
  - calls plot_distributions      : any extra per-variable comparison you pass
                                    via --variables

The signal dataset is identified by --signal-file, background by --bkg-file.

Usage:
    validate_ntuples.py -s <signal_file> -b <bkg_file> -o <output_dir>

Options:
    -s <signal_file>   Weighted signal parquet (e.g. z_train.parquet).
    -b <bkg_file>      Weighted background parquet (e.g. qq_train.parquet).
    -o <output_dir>    Directory where plots will be written.
"""

import os
import sys
import numpy as np
import awkward as ak
from pathlib import Path
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt
import general as g
import data_integrity as di


if __name__ == "__main__":
    args = docopt(__doc__)

    signal_file = args["-s"]
    bkg_file = args["-b"]
    output_dir = args["-o"]

    os.makedirs(output_dir, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    sig_data = ak.from_parquet(signal_file)
    bkg_data = ak.from_parquet(bkg_file)

    # ── extract per-event weights (added by apply_weights.py) ─────────────────
    sig_weights = np.asarray(ak.to_numpy(sig_data["cls_weight"]))
    bkg_weights = np.asarray(ak.to_numpy(bkg_data["cls_weight"]))

    # ── weighting overview plots (pt, eta, theta, p — weighted vs unweighted) ─
    wt.plot_weighting_results(
        signal_data=sig_data,
        bkg_data=bkg_data,
        sig_weights=sig_weights,
        bkg_weights=bkg_weights,
        output_dir=output_dir,
    )

    # ── additional per-variable distributions ─────────────────────────────────
    sig_p4s = g.reinitialize_p4(sig_data.gen_jet_p4)
    bkg_p4s = g.reinitialize_p4(bkg_data.gen_jet_p4)

    extra_vars = [
        (sig_p4s.mass, bkg_p4s.mass, r"$m$ [GeV]", "mass"),
        (
            ak.num(sig_data.reco_cand_p4s),
            ak.num(bkg_data.reco_cand_p4s),
            "N candidates",
            "n_cands",
        ),
    ]

    for sig_vals, bkg_vals, xlabel, name in extra_vars:
        sig_vals = np.asarray(ak.to_numpy(sig_vals))
        bkg_vals = np.asarray(ak.to_numpy(bkg_vals))
        wt.plot_distributions(
            sig_values=sig_vals,
            bkg_values=bkg_vals,
            sig_weights_unw=np.ones(len(sig_vals)) / len(sig_vals),
            bkg_weights_unw=np.ones(len(bkg_vals)) / len(bkg_vals),
            sig_weights_w=sig_weights / sig_weights.sum(),
            bkg_weights_w=bkg_weights / bkg_weights.sum(),
            output_path=os.path.join(output_dir, f"{name}.pdf"),
            xlabel=xlabel,
        )

    # ── data integrity plots (per-dataset) ────────────────────────────────────
    for label, data in [("signal", sig_data), ("background", bkg_data)]:
        di.plot_num_particles_per_jet(data, label=label).figure.savefig(
            os.path.join(output_dir, f"{label}_num_particles_per_jet.pdf"),
            bbox_inches="tight",
        )

        di.plot_jet_pt(data, label=label).figure.savefig(
            os.path.join(output_dir, f"{label}_reco_jet_pt.pdf"),
            bbox_inches="tight",
        )

        if label == "signal":
            di.plot_decay_mode_distribution(
                data, title=f"{label.capitalize()} decay modes"
            ).figure.savefig(
                os.path.join(output_dir, f"{label}_decay_modes.pdf"),
                bbox_inches="tight",
            )

        if "gen_jet_tau_vis_energy" in data.fields:
            di.plot_reco_jet_energy(data, label=label).figure.savefig(
                os.path.join(output_dir, f"{label}_reco_jet_energy.pdf"),
                bbox_inches="tight",
            )

        if "reco_cand_matched_gen_energy" in data.fields:
            di.plot_reco_vs_gen_cand_energy(data, label=label).figure.savefig(
                os.path.join(output_dir, f"{label}_reco_cand_energy.pdf"),
                bbox_inches="tight",
            )

        for var in ["reco_cand_dxy", "reco_cand_dz", "reco_cand_d3"]:
            if var in data.fields:
                di.plot_lifetime_variable(data, var, label=label).figure.savefig(
                    os.path.join(output_dir, f"{label}_{var}.pdf"),
                    bbox_inches="tight",
                )

    import matplotlib.pyplot as plt

    plt.close("all")

    print(f"Validation plots written to {output_dir}")
