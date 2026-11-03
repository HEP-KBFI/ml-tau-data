"""Produce validation plots for the weighted ntuple outputs.

Reads the weighted parquet chunks produced by the merge stage, then compares
all samples on the weight-matrix (theta, p) bins. Detailed plots compare
signal with each background separately.

Each background is identified by its short name and first train chunk.

Usage:
    validate_ntuples.py -s <signal_file> (-b <named_background>)... -w <weights_dir> -o <output_dir>

Options:
    -s <signal_file>   Weighted signal parquet (e.g. z_train_00000.parquet). Loaded
                       in full, so pass a single chunk rather than a whole split.
    -b <named_background> Background as <name>:<file>; repeat for every sample.
    -w <weights_dir>     Directory containing p_edges.npy and theta_edges.npy.
    -o <output_dir>    Directory where plots will be written.
"""

import os
import sys
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path
from docopt import docopt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import weight_tools as wt
import general as g
import data_integrity as di


def plot_all_sample_weights(samples, weights_dir, output_dir):
    theta_edges = np.load(os.path.join(weights_dir, "theta_edges.npy"))
    p_edges = np.load(os.path.join(weights_dir, "p_edges.npy"))
    histograms = {}
    for name, path in samples.items():
        data = ak.from_parquet(path, columns=["gen_jet_p4", "cls_weight"])
        theta, p = wt.gen_jet_theta_p(data)
        weights = np.asarray(ak.to_numpy(data["cls_weight"]))
        unweighted = np.histogram2d(theta, p, bins=(theta_edges, p_edges))[0]
        weighted = np.histogram2d(theta, p, bins=(theta_edges, p_edges), weights=weights)[0]
        if not unweighted.sum() or not weighted.sum():
            raise ValueError(f"No weighted jets in the configured (theta, p) range for {name}")
        histograms[name] = (unweighted / unweighted.sum(), weighted / weighted.sum())

    fig, axes = plt.subplots(
        2, len(samples), figsize=(4 * len(samples), 8), squeeze=False,
        constrained_layout=True,
    )
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("white")
    for row, title in enumerate(("Unweighted", "Weighted")):
        maximum = max(pair[row].max() for pair in histograms.values())
        for col, (name, pair) in enumerate(histograms.items()):
            ax = axes[row, col]
            ax.set_facecolor("white")
            image = ax.pcolormesh(
                p_edges, theta_edges, np.ma.masked_equal(pair[row], 0),
                cmap=colormap, vmin=0, vmax=maximum,
            )
            caption = f"{name} ({title})"
            if row == 1 and name != "signal":
                difference = 0.5 * np.abs(pair[1] - histograms["signal"][1]).sum()
                caption += f"\ndifference={difference:.3f}"
                print(f"Weighted 2D shape difference (signal vs {name}): {difference:.4f}")
            ax.set_title(caption, fontsize=11)
            ax.set_xlabel("p [GeV]", fontsize=10)
            ax.set_ylabel("theta [degrees]", fontsize=10)
            ax.tick_params(labelsize=9)
        colorbar = fig.colorbar(image, ax=axes[row, :].tolist())
        colorbar.set_label("Fraction / bin", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)
    fig.savefig(os.path.join(output_dir, "all_samples_theta_p.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for name, (_, weighted) in histograms.items():
        theta_projection = weighted.sum(axis=1)
        p_projection = weighted.sum(axis=0)
        axes[0].step(theta_edges, np.r_[theta_projection, 0], where="post", label=name)
        axes[1].step(p_edges, np.r_[p_projection, 0], where="post", label=name)
    for ax, xlabel in zip(axes, ("theta [degrees]", "p [GeV]")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Weighted fraction / bin")
        ax.legend(facecolor="white", framealpha=1, frameon=True)
    fig.savefig(os.path.join(output_dir, "all_samples_weighted_projections.pdf"))
    plt.close(fig)


def plot_pair(sig_data, bkg_data, output_dir, background_name):
    background_label = f"Background ({background_name})"
    # ── extract per-event weights (added by the merge stage) ──────────────────
    sig_weights = np.asarray(ak.to_numpy(sig_data["cls_weight"]))
    bkg_weights = np.asarray(ak.to_numpy(bkg_data["cls_weight"]))

    # ── weighting overview plots (pt, eta, theta, p — weighted vs unweighted) ─
    wt.plot_weighting_results(
        signal_data=sig_data,
        bkg_data=bkg_data,
        sig_weights=sig_weights,
        bkg_weights=bkg_weights,
        output_dir=output_dir,
        bkg_label=background_label,
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
            bkg_label=background_label,
        )

    # ── data integrity plots (signal and background overlaid) ─────────────────
    def save_overlay(plot_fn, output_name, is_2d=False, **kwargs):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        if is_2d:
            plot_fn(sig_data, label=f"Signal only (paired with {background_name})", ax=ax, **kwargs)
        else:
            plot_fn(sig_data, label="Signal", ax=ax, color="red", **kwargs)
            plot_fn(bkg_data, label=background_label, ax=ax, color="blue", **kwargs)
            ax.legend(facecolor="white", framealpha=1, frameon=True)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, output_name), bbox_inches="tight")
        plt.close(fig)

    save_overlay(di.plot_num_particles_per_jet, "num_particles_per_jet.pdf")
    save_overlay(di.plot_jet_pt, "reco_jet_pt.pdf")

    if "gen_jet_tau_vis_energy" in sig_data.fields:
        save_overlay(di.plot_reco_jet_energy, "reco_jet_energy.pdf", is_2d=True)

    if "reco_cand_matched_gen_energy" in sig_data.fields:
        save_overlay(di.plot_reco_vs_gen_cand_energy, "reco_cand_energy.pdf", is_2d=True)

    _lifetime_bins = {
        "reco_cand_dxy": np.logspace(-3, 1, 80),   # 1 µm – 10 mm
        "reco_cand_dz":  np.logspace(-3, 1, 80),   # 1 µm – 10 mm
        "reco_cand_d3":  np.logspace(-3, 2, 80),   # 1 µm – 100 mm (K0s/Lambda tail)
    }
    for var in ["reco_cand_dxy", "reco_cand_dz", "reco_cand_d3"]:
        if var in sig_data.fields:
            log_bins = _lifetime_bins[var]
            fig, ax = plt.subplots(figsize=(7, 5.5))
            di.plot_lifetime_variable(
                sig_data,
                label="Signal",
                ax=ax,
                color="red",
                variable=var,
                bins=log_bins,
            )
            di.plot_lifetime_variable(
                bkg_data,
                label=background_label,
                ax=ax,
                color="blue",
                variable=var,
                bins=log_bins,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(facecolor="white", framealpha=1, frameon=True)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{var}.pdf"), bbox_inches="tight")
            plt.close(fig)

    if "gen_jet_tau_decaymode" in sig_data.fields:
        di.plot_decay_mode_distribution(
            sig_data, title=f"Signal decay modes (paired with {background_name})"
        ).figure.savefig(
            os.path.join(output_dir, "signal_decay_modes.pdf"), bbox_inches="tight"
        )

    # ── weight distributions and impact-parameter error overlays ─────────────
    # These used to be produced by apply_weights.py -p; the weights are now
    # applied by the merge stage, so the plots live with the other validation.
    wt.plot_weight_distributions(sig_weights, bkg_weights, output_dir, bkg_label=background_label)

    INVALID = 0.0  # lifetime.INVALID_VALUE
    error_vars = {
        "reco_cand_dxy_error": "PFCandidate dxy error [mm]",
        "reco_cand_dz_error": "PFCandidate dz error [mm]",
    }
    error_bins = np.logspace(-4, 0, 80)
    for var, xlabel in error_vars.items():
        if var not in sig_data.fields:
            continue
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for data, label, color in [
            (sig_data, "Signal", "red"),
            (bkg_data, background_label, "blue"),
        ]:
            flat = ak.to_numpy(ak.flatten(data[var]))
            flat = flat[flat > INVALID]  # errors of real tracks are > 0
            counts, edges = np.histogram(flat, bins=error_bins, density=True)
            ax.step(edges[:-1], counts, where="post", label=label, color=color)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction [a.u.]")
        ax.legend(facecolor="white", framealpha=1, frameon=True)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{var}.pdf"), bbox_inches="tight")
        plt.close(fig)

    plt.close("all")

    print(f"Validation plots written to {output_dir}")


if __name__ == "__main__":
    args = docopt(__doc__)

    signal_file = args["-s"]
    bkg_files = dict(value.split(":", 1) for value in args["-b"])
    weights_dir = args["-w"]
    output_dir = args["-o"]

    os.makedirs(output_dir, exist_ok=True)
    plot_all_sample_weights({"signal": signal_file, **bkg_files}, weights_dir, output_dir)

    sig_data = ak.from_parquet(signal_file)
    for name, bkg_file in bkg_files.items():
        pair_dir = os.path.join(output_dir, f"signal_vs_{name}")
        os.makedirs(pair_dir, exist_ok=True)
        plot_pair(sig_data, ak.from_parquet(bkg_file), pair_dir, name)
