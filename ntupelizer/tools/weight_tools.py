import os
import glob
import hydra
import matplotlib
import numpy as np
import general as g
import mplhep as hep
import awkward as ak
import seaborn as sns
import multiprocessing
import matplotlib as mpl
from itertools import repeat
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from general import load_all_data
from matplotlib import ticker
from matplotlib.ticker import AutoLocator

hep.style.use(hep.styles.CMS)
matplotlib.use("Agg")


def load_samples(sig_dir: str, bkg_dir: str, n_files: int = -1, branches: list = None):
    sig_data_ = load_all_data(sig_dir, n_files=n_files, branches=branches)
    bkg_data = load_all_data(bkg_dir, n_files=n_files, branches=branches)
    sig_data = sig_data_[sig_data_.gen_jet_tau_decaymode != -1]
    return sig_data, bkg_data


def visualize_weights(
    weight_matrix,
    x_bin_edges,
    y_bin_edges,
    output_path,
    ylabel=r"$\eta$",
    xlabel=r"$p_T$",
):
    x_labels = [f"{label:9.0f}" for label in (x_bin_edges[1:] + x_bin_edges[:-1]) / 2]
    y_labels = [f"{label:9.2f}" for label in (y_bin_edges[1:] + y_bin_edges[:-1]) / 2]
    hep.style.use(hep.styles.CMS)
    sns.set_style(rc={"figure.figsize": (16, 9)})
    heatmap = sns.heatmap(weight_matrix)
    heatmap.set_xticks(range(len(x_labels)))
    heatmap.set_xticklabels(x_labels)
    heatmap.set_yticks(range(len(y_labels)))
    heatmap.set_yticklabels(y_labels)
    plt.ylabel(ylabel)
    plt.yticks(rotation=0)
    plt.xlabel(xlabel)
    heatmap.yaxis.set_major_locator(AutoLocator())
    heatmap.xaxis.set_major_locator(AutoLocator())
    plt.savefig(output_path)
    plt.close("all")


def create_matrix(data, y_bin_edges, x_bin_edges, y_property, x_property):
    p4s = g.reinitialize_p4(data.gen_jet_p4)
    x_property_ = getattr(p4s, x_property).to_numpy()
    y_property_ = getattr(p4s, y_property).to_numpy()
    if y_property == "theta":
        y_property_ = np.rad2deg(y_property_)
    matrix = np.histogram2d(y_property_, x_property_, bins=(y_bin_edges, x_bin_edges))[
        0
    ]
    normalized_matrix = matrix / np.sum(matrix)
    return normalized_matrix


def get_weight_matrix(target_matrix, comp_matrix):
    weights = np.minimum(target_matrix, comp_matrix) / target_matrix
    return np.nan_to_num(weights, nan=0.0)


def process_files(
    weight_matrix,
    theta_bin_edges,
    pt_bin_edges,
    data_dir,
    cfg,
    use_multiprocessing=True,
):
    data_paths = glob.glob(os.path.join(data_dir, "*.parquet"))
    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=8)
        weights = []
        for result in pool.starmap(
            process_single_file,
            zip(
                data_paths,
                repeat(weight_matrix),
                repeat(theta_bin_edges),
                repeat(pt_bin_edges),
                repeat(cfg),
            ),
        ):
            weights.extend(result)
    else:
        for input_path in data_paths:
            weights.extend(
                process_single_file(
                    input_path=input_path,
                    weight_matrix=weight_matrix,
                    theta_bin_edges=theta_bin_edges,
                    pt_bin_edges=pt_bin_edges,
                    cfg=cfg,
                )
            )
    return weights


def get_weights(data, weight_matrix, theta_bin_edges, pt_bin_edges):
    p4s = g.reinitialize_p4(data.gen_jet_p4)
    theta_values = np.rad2deg(p4s.theta.to_numpy())
    pt_values = p4s.p.to_numpy()
    theta_bin = (
        np.digitize(theta_values, bins=(theta_bin_edges[1:] + theta_bin_edges[:-1]) / 2)
        - 1
    )
    pt_bin = np.digitize(pt_values, bins=(pt_bin_edges[1:] + pt_bin_edges[:-1]) / 2) - 1
    matrix_loc = np.concatenate(
        [theta_bin.reshape(-1, 1), pt_bin.reshape(-1, 1)], axis=1
    )
    weights = ak.from_iter([weight_matrix[tuple(loc)] for loc in matrix_loc])
    return weights


def process_single_file(input_path, weight_matrix, theta_bin_edges, pt_bin_edges, cfg):
    data = ak.from_parquet(input_path)
    weights = get_weights(data, weight_matrix, theta_bin_edges, pt_bin_edges)
    if cfg.add_weights:
        merged_info = {field: data[field] for field in data.fields}
        merged_info.update({"weight": weights})
        print(f"Adding weights to {input_path}")
        ak.to_parquet(ak.Record(merged_info), input_path)
    return weights


def plot_weighting_results(signal_data, bkg_data, sig_weights, bkg_weights, output_dir):
    tau_mask = signal_data.gen_jet_tau_decaymode != -1
    sig_data = signal_data[tau_mask]
    sig_weights = np.asarray(sig_weights)[np.asarray(tau_mask)]
    # ZH_bkg = signal_data[signal_data.gen_jet_tau_decaymode == -1]
    # bkg_data = ak.concatenate([bkg_data, ZH_bkg], axis=0) # Maybe should at somepoint also think about including
    # bkg jets from signal sample
    bkg_p4s = g.reinitialize_p4(bkg_data.gen_jet_p4)
    sig_p4s = g.reinitialize_p4(sig_data.gen_jet_p4)

    variables = [
        (sig_p4s.pt, bkg_p4s.pt, r"$p_T$ [GeV]", "pt", True, 30),
        (sig_p4s.eta, bkg_p4s.eta, r"$\eta$", "eta", False, None),
        (sig_p4s.p, bkg_p4s.p, r"$p$ [GeV]", "p", True, 30),
        (
            np.rad2deg(sig_p4s.theta.to_numpy()),
            np.rad2deg(bkg_p4s.theta.to_numpy()),
            r"$\theta$ [$^{o}$]",
            "theta",
            False,
            50,
        ),
    ]
    for sig_vals, bkg_vals, xlabel, name, produce_label, x_tick in variables:
        plot_distributions(
            sig_values=sig_vals,
            bkg_values=bkg_vals,
            sig_weights_unw=np.ones(len(sig_p4s.pt)) / len(sig_p4s.pt),
            bkg_weights_unw=np.ones(len(bkg_p4s.pt)) / len(bkg_p4s.pt),
            sig_weights_w=sig_weights / sig_weights.sum(),
            bkg_weights_w=bkg_weights / bkg_weights.sum(),
            output_path=os.path.join(output_dir, f"{name}_normalized.pdf"),
            xlabel=xlabel,
            produce_label=produce_label,
            x_maj_tick_spacing=x_tick,
        )


def plot_distributions(
    sig_values,
    bkg_values,
    sig_weights_unw,
    bkg_weights_unw,
    sig_weights_w,
    bkg_weights_w,
    output_path,
    xlabel=r"$p_T [GeV]$",
    produce_label=True,
    x_maj_tick_spacing=30,
):
    mpl.rcParams.update(mpl.rcParamsDefault)
    hep.style.use(hep.styles.CMS)
    # Use the same bin edges for both panels so they are directly comparable.
    _, bin_edges = np.histogram(bkg_values, bins=50)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7), sharey=False)

    for ax, (sw, bw, title) in zip(
        axes,
        [
            (sig_weights_unw, bkg_weights_unw, "Unweighted"),
            (sig_weights_w, bkg_weights_w, "Weighted"),
        ],
    ):
        sig_hist = np.histogram(sig_values, weights=sw, bins=bin_edges)[0]
        bkg_hist = np.histogram(bkg_values, weights=bw, bins=bin_edges)[0]
        hep.histplot(
            sig_hist,
            bins=bin_edges,
            histtype="step",
            label="Signal",
            hatch="\\\\",
            color="red",
            ax=ax,
        )
        hep.histplot(
            bkg_hist,
            bins=bin_edges,
            histtype="step",
            label="Background",
            hatch="//",
            color="blue",
            ax=ax,
        )
        ax.set_facecolor("white")
        ax.set_xlabel(xlabel, fontsize=26)
        ax.set_ylabel("Relative yield / bin", fontsize=26)
        ax.tick_params(axis="x", labelsize=22)
        ax.tick_params(axis="y", labelsize=22)
        ax.set_title(title, fontsize=26)
        ax.yaxis.set_major_locator(AutoLocator())
        if x_maj_tick_spacing is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_maj_tick_spacing))
        if produce_label:
            ax.legend(fontsize=20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


@hydra.main(config_path="../config", config_name="weighting", version_base=None)
def main(cfg: DictConfig):
    sig_data, bkg_data = load_samples(
        sig_dir=cfg.samples.ZH_Htautau.output_dir,
        bkg_dir=cfg.samples.QCD.output_dir,
        n_files=cfg.n_files_per_sample,
        branches=["gen_jet_p4", "gen_jet_tau_decaymode"],
    )
    output_dir = os.path.abspath(os.path.join(cfg.samples.QCD.output_dir, os.pardir))
    eta_bin_edges = np.linspace(
        cfg.weighting.variables.eta.range[0],
        cfg.weighting.variables.eta.range[1],
        cfg.weighting.variables.eta.n_bins,
    )
    theta_bin_edges = np.linspace(
        cfg.weighting.variables.theta.range[0],
        cfg.weighting.variables.theta.range[1],
        cfg.weighting.variables.theta.n_bins,
    )
    pt_bin_edges = np.linspace(
        cfg.weighting.variables.pt.range[0],
        cfg.weighting.variables.pt.range[1],
        cfg.weighting.variables.pt.n_bins,
    )
    sig_matrix = create_matrix(
        sig_data, eta_bin_edges, pt_bin_edges, y_property="eta", x_property="pt"
    )
    bkg_matrix = create_matrix(
        bkg_data, eta_bin_edges, pt_bin_edges, y_property="eta", x_property="pt"
    )
    if cfg.produce_plots:
        sig_output_path = os.path.join(output_dir, "signal_matrix.pdf")
        visualize_weights(sig_matrix, pt_bin_edges, eta_bin_edges, sig_output_path)
        bkg_output_path = os.path.join(output_dir, "bkg_matrix.pdf")
        visualize_weights(bkg_matrix, pt_bin_edges, eta_bin_edges, bkg_output_path)
        total_output_path = os.path.join(output_dir, "total_matrix.pdf")
        visualize_weights(
            sig_matrix + bkg_matrix, pt_bin_edges, eta_bin_edges, total_output_path
        )
        normed_total_output_path = os.path.join(output_dir, "normed_total_matrix.pdf")
        visualize_weights(
            (sig_matrix + bkg_matrix) / (np.sum(sig_matrix) + np.sum(bkg_matrix)),
            pt_bin_edges,
            eta_bin_edges,
            normed_total_output_path,
        )
    sig_weights = get_weight_matrix(target_matrix=sig_matrix, comp_matrix=bkg_matrix)
    bkg_weights = get_weight_matrix(target_matrix=bkg_matrix, comp_matrix=sig_matrix)
    if cfg.produce_plots:
        sig_output_path = os.path.join(output_dir, "signal_weights.pdf")
        visualize_weights(sig_weights, pt_bin_edges, eta_bin_edges, sig_output_path)
        bkg_output_path = os.path.join(output_dir, "bkg_weights.pdf")
        visualize_weights(bkg_weights, pt_bin_edges, eta_bin_edges, bkg_output_path)

    sig_matrix_p_theta = create_matrix(
        sig_data, theta_bin_edges, pt_bin_edges, y_property="theta", x_property="p"
    )
    bkg_matrix_p_theta = create_matrix(
        bkg_data, theta_bin_edges, pt_bin_edges, y_property="theta", x_property="p"
    )
    if cfg.produce_plots:
        print("Visualizing distributions")
        sig_output_path_p_theta = os.path.join(output_dir, "signal_matrix_p_theta.pdf")
        visualize_weights(
            sig_matrix_p_theta,
            pt_bin_edges,
            theta_bin_edges,
            sig_output_path_p_theta,
            ylabel=r"$\theta$",
            xlabel="p",
        )
        bkg_output_path_p_theta = os.path.join(output_dir, "bkg_matrix_p_theta.pdf")
        visualize_weights(
            bkg_matrix_p_theta,
            pt_bin_edges,
            theta_bin_edges,
            bkg_output_path_p_theta,
            ylabel=r"$\theta$",
            xlabel="p",
        )
        total_output_path_p_theta = os.path.join(output_dir, "total_matrix_p_theta.pdf")
        visualize_weights(
            sig_matrix_p_theta + bkg_matrix_p_theta,
            pt_bin_edges,
            theta_bin_edges,
            total_output_path_p_theta,
            ylabel=r"$\theta$",
            xlabel="p",
        )
        normed_total_output_path_p_theta = os.path.join(
            output_dir, "normed_total_matrix_p_theta.pdf"
        )
        visualize_weights(
            (sig_matrix_p_theta + bkg_matrix_p_theta)
            / (np.sum(sig_matrix_p_theta) + np.sum(bkg_matrix_p_theta)),
            pt_bin_edges,
            theta_bin_edges,
            normed_total_output_path_p_theta,
            ylabel=r"$\theta$",
            xlabel="p",
        )
    sig_weights_p_theta = get_weight_matrix(
        target_matrix=sig_matrix_p_theta, comp_matrix=bkg_matrix_p_theta
    )
    bkg_weights_p_theta = get_weight_matrix(
        target_matrix=bkg_matrix_p_theta, comp_matrix=sig_matrix_p_theta
    )
    signal_weights = process_files(
        weight_matrix=sig_weights_p_theta,
        theta_bin_edges=theta_bin_edges,
        pt_bin_edges=pt_bin_edges,
        data_dir=cfg.samples.ZH_Htautau.output_dir,
        cfg=cfg,
        use_multiprocessing=cfg.use_multiprocessing,
    )
    bkg_weights = process_files(
        weight_matrix=bkg_weights_p_theta,
        theta_bin_edges=theta_bin_edges,
        pt_bin_edges=pt_bin_edges,
        data_dir=cfg.samples.QCD.output_dir,
        cfg=cfg,
        use_multiprocessing=cfg.use_multiprocessing,
    )
    plot_weight_distributions(signal_weights, bkg_weights, output_dir)
    if cfg.produce_plots:
        print("Visualizing weights and plotting the weighting results")
        sig_output_path_p_theta = os.path.join(output_dir, "signal_weights_p_theta.pdf")
        visualize_weights(
            weight_matrix=sig_weights_p_theta,
            x_bin_edges=pt_bin_edges,
            y_bin_edges=theta_bin_edges,
            output_path=sig_output_path_p_theta,
            ylabel=r"$\theta$",
            xlabel="p",
        )
        bkg_output_path_p_theta = os.path.join(output_dir, "bkg_weights_p_theta.pdf")
        visualize_weights(
            weight_matrix=bkg_weights_p_theta,
            x_bin_edges=pt_bin_edges,
            y_bin_edges=theta_bin_edges,
            output_path=bkg_output_path_p_theta,
            ylabel=r"$\theta$",
            xlabel="p",
        )
        sig_weights = get_weights(
            sig_data, sig_weights_p_theta, theta_bin_edges, pt_bin_edges
        )
        bkg_weights = get_weights(
            bkg_data, bkg_weights_p_theta, theta_bin_edges, pt_bin_edges
        )
        plot_weighting_results(
            sig_data,
            bkg_data,
            sig_weights=sig_weights,
            bkg_weights=bkg_weights,
            output_dir=output_dir,
        )


def plot_weight_distributions(signal_weights, bkg_weights, output_dir):
    mpl.rcParams.update(mpl.rcParamsDefault)
    hep.style.use(hep.styles.CMS)
    bin_edges = np.linspace(start=0, stop=1, num=51)
    bkg_hist_ = np.histogram(bkg_weights, bins=bin_edges)[0]
    bkg_hist = bkg_hist_ / np.sum(bkg_hist_)
    sig_hist_ = np.histogram(signal_weights, bins=bin_edges)[0]
    sig_hist = sig_hist_ / np.sum(sig_hist_)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    hep.histplot(bkg_hist, bin_edges, label="Quark/gluon jets", hatch="//", color="red")
    hep.histplot(sig_hist, bin_edges, label=r"$\tau_h$", hatch="\\\\", color="blue")
    plt.xlabel("Weight", fontdict={"size": 30})
    plt.ylabel("Relative yield / bin", fontdict={"size": 30})
    plt.legend()
    ax.set_facecolor("white")
    output_path = os.path.join(output_dir, "weight_1D_distribution.pdf")
    plt.savefig(output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
