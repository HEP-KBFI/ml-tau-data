"""
Data integrity and validation functions for ntupelizer output.

This module provides functions to visualize and validate the output
from the ntupelizer.py ntupelize function.
"""

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from typing import Optional, Dict

from general import reinitialize_p4, to_bh, deltaphi

try:
    import mplhep

    mplhep.style.use("CMS")
except ImportError:
    print("mplhep not installed, using default matplotlib style")


# =============================================================================
# Decay mode utilities
# =============================================================================


REMAP_LABELS = {
    0: r"$h^\pm$",
    1: r"$h^\pm \pi^0$",
    2: r"$h^\pm \geq 2\pi^0$",
    3: r"$h^\pm h^\mp h^\pm$",
    4: r"$h^\pm h^\mp h^\pm \geq 1\pi^0$",
    5: "Other",
}


def remap_decaymodes(dm_array: ak.Array) -> np.ndarray:
    """Remap decay modes to reduced set.

    Mapping:
        0 -> 0 (h±)
        1 -> 1 (h± π0)
        2, 3, 4 -> 2 (h± ≥2π0)
        10 -> 3 (3h±)
        11, 12, 13, 14 -> 4 (3h± ≥1π0)
        5-9, 15 -> 5 (Other)

    Args:
        dm_array: Original decay mode array

    Returns:
        Remapped decay mode array
    """
    dm_array = ak.to_numpy(dm_array)
    new_array = np.ones(len(dm_array)) * -1

    new_array[dm_array == 0] = 0
    new_array[dm_array == 1] = 1
    new_array[dm_array == 2] = 2
    new_array[dm_array == 3] = 2
    new_array[dm_array == 4] = 2
    new_array[dm_array == 10] = 3
    new_array[dm_array == 11] = 4
    new_array[dm_array == 12] = 4
    new_array[dm_array == 13] = 4
    new_array[dm_array == 14] = 4
    new_array[dm_array == 15] = 5
    new_array[dm_array == 5] = 5
    new_array[dm_array == 6] = 5
    new_array[dm_array == 7] = 5
    new_array[dm_array == 8] = 5
    new_array[dm_array == 9] = 5

    if np.any(new_array == -1):
        print("Warning: Array contains unmapped decay modes (-1)")

    return new_array


def dm_percentages(data: ak.Array) -> Dict[int, float]:
    """Calculate decay mode percentages.

    Args:
        data: Decay mode array

    Returns:
        Dictionary mapping decay mode to percentage
    """
    uniques, counts = np.unique(data, return_counts=True)
    total_count = sum(counts)
    percentages = np.round((counts / total_count) * 100, 2)
    return dict(zip(uniques, percentages))


# =============================================================================
# Single dataset plotting functions
# =============================================================================


def plot_variable_distribution(
    data: ak.Array,
    variable: str,
    bins: np.ndarray,
    label: str = "Data",
    xlabel: Optional[str] = None,
    ylabel: str = "Fraction [a.u.]",
    log_scale: bool = True,
    density: bool = True,
    flatten: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot distribution of a single variable.

    Args:
        data: Ntupelizer output data
        variable: Variable name to plot
        bins: Bin edges
        label: Label for legend
        xlabel: X-axis label (defaults to variable name)
        ylabel: Y-axis label
        log_scale: Use log scale for y-axis
        density: Normalize histogram
        flatten: Flatten jagged arrays
        ax: Matplotlib axes (creates new if None)
        **kwargs: Additional arguments passed to mplhep.histplot

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))

    var_data = data[variable]
    if flatten:
        var_data = ak.flatten(var_data)

    try:
        mplhep.histplot(
            to_bh(var_data, bins=bins),
            histtype="step",
            lw=1,
            flow="sum",
            label=label,
            density=density,
            ax=ax,
            **kwargs,
        )
    except NameError:
        ax.hist(
            ak.to_numpy(var_data),
            bins=bins,
            histtype="step",
            label=label,
            density=density,
            **kwargs,
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or variable)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")

    return ax


def plot_jet_pt(
    data: ak.Array,
    label: str = "Data",
    bins: Optional[np.ndarray] = None,
    decaymode_mask: Optional[ak.Array] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot reconstructed jet pT distribution.

    Args:
        data: Ntupelizer output data
        label: Label for legend
        bins: Bin edges (default: 0-220 GeV in 51 bins)
        decaymode_mask: Optional mask to select specific decay modes
        ax: Matplotlib axes
        **kwargs: Additional arguments passed to plot

    Returns:
        Matplotlib axes object
    """
    if bins is None:
        bins = np.linspace(0, 220, 51)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))

    jet_pt = reinitialize_p4(data["reco_jet_p4"]).pt
    if decaymode_mask is not None:
        jet_pt = jet_pt[decaymode_mask]

    try:
        mplhep.histplot(
            to_bh(jet_pt, bins=bins),
            histtype="step",
            lw=1,
            flow="sum",
            label=label,
            density=True,
            ax=ax,
            **kwargs,
        )
    except NameError:
        ax.hist(
            ak.to_numpy(jet_pt),
            bins=bins,
            histtype="step",
            label=label,
            density=True,
            **kwargs,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Reco jet $p_T$ [GeV]")
    ax.set_ylabel("Fraction of reco jets [a.u.]")
    ax.legend(loc="best")

    return ax


def plot_num_particles_per_jet(
    data: ak.Array,
    label: str = "Data",
    bins: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot number of particles per jet distribution.

    Args:
        data: Ntupelizer output data
        label: Label for legend
        bins: Bin edges (default: 0-50)
        ax: Matplotlib axes
        **kwargs: Additional arguments

    Returns:
        Matplotlib axes object
    """
    if bins is None:
        bins = np.linspace(0, 50, 51)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))

    num_particles = ak.num(data["reco_cand_p4s"])

    try:
        mplhep.histplot(
            to_bh(num_particles, bins=bins),
            histtype="step",
            lw=1,
            flow="sum",
            label=label,
            density=True,
            ax=ax,
            **kwargs,
        )
    except NameError:
        ax.hist(
            ak.to_numpy(num_particles),
            bins=bins,
            histtype="step",
            label=label,
            density=True,
            **kwargs,
        )

    ax.set_xlabel("Number of reco particles / jet")
    ax.set_ylabel("Fraction of reco jets [a.u.]")
    ax.legend(loc="best")

    return ax


def plot_decay_mode_distribution(
    data: ak.Array,
    ax: Optional[plt.Axes] = None,
    title: str = "Decay modes",
    annotate: bool = True,
) -> plt.Axes:
    """Plot decay mode distribution with percentages.

    Args:
        data: Ntupelizer output data
        ax: Matplotlib axes
        title: Plot title
        annotate: Add percentage annotations

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))

    dm_data = data["gen_jet_tau_decaymode"]
    percentages = dm_percentages(dm_data)
    dms = np.arange(17)

    counts, _, bars = ax.hist(
        ak.to_numpy(dm_data), bins=dms, width=1, edgecolor="black", linewidth=0.5
    )

    ax.set_yscale("log")
    ax.set_xticks(dms + 0.5, dms)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Decay mode")
    ax.set_ylabel("Number of jets")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    if annotate:
        for bar in ax.patches:
            height = bar.get_height()
            decay_mode = int(bar.get_x() + bar.get_width() / 2)
            percentage = percentages.get(decay_mode, 0)
            if height > 0:
                ax.annotate(
                    f"{percentage:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="red",
                    rotation=0,
                )

    return ax


def plot_2d_jet_shape(
    data: ak.Array,
    decaymode_mask: Optional[ak.Array] = None,
    title: str = "Jet shape",
    bins: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot 2D jet shape (Δη vs Δφ weighted by pT).

    Args:
        data: Ntupelizer output data
        decaymode_mask: Optional mask to select specific decay modes
        title: Plot title
        bins: Bin edges for both axes
        ax: Matplotlib axes
        save_path: Path to save figure

    Returns:
        Matplotlib axes object
    """
    if bins is None:
        bins = np.linspace(-0.05, 0.05, 70)

    reco_jet_p4s = reinitialize_p4(data["reco_jet_p4"])
    reco_cand_p4s = reinitialize_p4(data["reco_cand_p4s"])

    delta_eta = reco_jet_p4s.eta - reco_cand_p4s.eta
    delta_phi = deltaphi(reco_jet_p4s.phi, reco_cand_p4s.phi)
    pt = reco_cand_p4s.pt

    if decaymode_mask is not None:
        delta_eta = delta_eta[decaymode_mask]
        delta_phi = delta_phi[decaymode_mask]
        pt = pt[decaymode_mask]

    delta_eta_flat = ak.to_numpy(ak.flatten(delta_eta))
    delta_phi_flat = ak.to_numpy(ak.flatten(delta_phi))
    pt_flat = ak.to_numpy(ak.flatten(pt))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    hist2d = ax.hist2d(
        delta_eta_flat,
        delta_phi_flat,
        bins=bins,
        weights=pt_flat / len(pt_flat),
        norm=mpl.colors.LogNorm(vmin=0.0001, vmax=10, clip=True),
        cmap=mpl.cm.jet,
        edgecolor="face",
    )

    ax.set_xlabel(r"$\Delta \eta$", fontsize=12)
    ax.set_ylabel(r"$\Delta \phi$", fontsize=12)
    ax.set_title(title, fontsize=12, pad=3)
    ax.set_aspect("equal")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    cbar = plt.colorbar(hist2d[3], ax=ax)
    cbar.set_label(r"Average $p_T$ [GeV]", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    return ax


def plot_jet_shapes_for_decay_modes(
    data: ak.Array,
    decay_modes_of_interest: list = None,
    dataset_name: str = "Data",
    output_dir: Optional[str] = None,
    show_plots: bool = True,
) -> Dict[int, plt.Axes]:
    """Plot 2D jet shapes for multiple decay modes.

    Args:
        data: Ntupelizer output data
        decay_modes_of_interest: List of remapped decay mode indices to plot.
            Default: [0, 1, 2, 3, 4] (all main decay modes)
        dataset_name: Name of the dataset for titles (e.g., "Z", "ZH")
        output_dir: Directory to save figures (None to skip saving)
        show_plots: Whether to display plots

    Returns:
        Dictionary mapping decay mode to axes object
    """
    if decay_modes_of_interest is None:
        decay_modes_of_interest = [0, 1, 2, 3, 4]

    dataset_labels = {
        "Z": r"$Z/\gamma \rightarrow \tau\tau$",
        "ZH": r"$ZH \rightarrow Z\tau\tau$",
    }

    dm_labels = dict(REMAP_LABELS)
    dm_labels[0] = r"$\tau_{h1}$"
    dm_labels[3] = r"$\tau_{h3}$"

    reco_jet_p4s = reinitialize_p4(data["reco_jet_p4"])
    reco_cand_p4s = reinitialize_p4(data["reco_cand_p4s"])

    delta_eta = reco_jet_p4s.eta - reco_cand_p4s.eta
    delta_phi = deltaphi(reco_jet_p4s.phi, reco_cand_p4s.phi)
    pt = reco_cand_p4s.pt

    remapped_dm = remap_decaymodes(data["gen_jet_tau_decaymode"])

    axes = {}
    bins = np.linspace(-0.05, 0.05, 70)

    for idx_dm in decay_modes_of_interest:
        mask = remapped_dm == idx_dm

        if np.sum(mask) == 0:
            print(f"Warning: No jets found for decay mode {idx_dm}")
            continue

        dm_label = dm_labels.get(idx_dm, f"DM{idx_dm}")
        dataset_label = dataset_labels.get(dataset_name, dataset_name)
        title = f"{dm_label} from {dataset_label}"

        delta_eta_flat = ak.to_numpy(ak.flatten(delta_eta[mask]))
        delta_phi_flat = ak.to_numpy(ak.flatten(delta_phi[mask]))
        pt_flat = ak.to_numpy(ak.flatten(pt[mask]))

        fig, ax = plt.subplots(figsize=(5, 5))

        hist2d = ax.hist2d(
            delta_eta_flat,
            delta_phi_flat,
            bins=bins,
            weights=pt_flat / len(pt_flat),
            norm=mpl.colors.LogNorm(vmin=0.0001, vmax=10, clip=True),
            cmap=mpl.cm.jet,
            edgecolor="face",
        )

        ax.set_xlabel(r"$\Delta \eta$", fontsize=12)
        ax.set_ylabel(r"$\Delta \phi$", fontsize=12)
        ax.set_title(title, fontsize=12, pad=3)
        ax.set_aspect("equal")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        cbar = plt.colorbar(hist2d[3], ax=ax)
        cbar.set_label(r"Average $p_T$ [GeV]", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        if output_dir:
            filename = f"{output_dir}/jet_2D_shapes_{dataset_name}_DM{idx_dm}.pdf"
            plt.savefig(filename, format="pdf", bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        axes[idx_dm] = ax

    return axes


def plot_lifetime_variable(
    data: ak.Array,
    variable: str,
    bins: Optional[np.ndarray] = None,
    label: str = "Data",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot lifetime variable distribution (dxy, dz, etc.).

    Args:
        data: Ntupelizer output data
        variable: Variable name (e.g., 'reco_cand_dxy', 'reco_cand_dz')
        bins: Bin edges
        label: Label for legend
        ax: Matplotlib axes
        **kwargs: Additional arguments

    Returns:
        Matplotlib axes object
    """
    if bins is None:
        bins = np.linspace(0, 5, 100)

    xlabel_map = {
        "reco_cand_dxy": "PFCandidate dxy [mm]",
        "reco_cand_dz": "PFCandidate dz [mm]",
        "reco_cand_dxy_err": "PFCandidate dxy error [mm]",
        "reco_cand_dz_err": "PFCandidate dz error [mm]",
        "reco_cand_d3": "PFCandidate d3 [mm]",
        "reco_cand_d0": "PFCandidate d0 [mm]",
        "reco_cand_z0": "PFCandidate z0 [mm]",
    }

    return plot_variable_distribution(
        data,
        variable,
        bins,
        label=label,
        xlabel=xlabel_map.get(variable, variable),
        ylabel="Fraction of reco particles [a.u.]",
        ax=ax,
        **kwargs,
    )


def plot_reco_jet_energy(
    data: ak.Array,
    label: str = "Data",
    bins: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot 2D gen vs reco jet energy.

    Args:
        data: Ntupelizer output data
        label: Dataset label for title
        bins: Bin edges for both axes
        ax: Matplotlib axes
        save_path: Path to save figure

    Returns:
        Matplotlib axes object
    """
    gen_jet_en = ak.to_numpy(data["gen_jet_tau_vis_energy"])
    reco_jet_p4 = reinitialize_p4(data["reco_jet_p4"])
    reco_jet_en = ak.to_numpy(reco_jet_p4.energy)

    valid = gen_jet_en != -1
    gen_jet_en = gen_jet_en[valid]
    reco_jet_en = reco_jet_en[valid]

    if bins is None:
        en_max = max(gen_jet_en.max(), reco_jet_en.max())
        bins = np.linspace(0, en_max, 51)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    else:
        fig = ax.figure

    hist2d = ax.hist2d(
        gen_jet_en,
        reco_jet_en,
        bins=bins,
        norm=mpl.colors.LogNorm(),
        cmap="viridis",
    )

    ax.set_xlabel("Gen visible tau energy [GeV]")
    ax.set_ylabel("Reco jet energy [GeV]")
    ax.set_title(f"{label} jet energy", fontsize=12)

    cbar = plt.colorbar(hist2d[3], ax=ax)
    cbar.set_label("Number of jets")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return ax


def plot_reco_vs_gen_cand_energy(
    data: ak.Array,
    label: str = "Data",
    bins: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot 2D gen vs reco candidate energy.

    Args:
        data: Ntupelizer output data
        label: Dataset label for title
        bins: Bin edges for both axes
        ax: Matplotlib axes
        save_path: Path to save figure

    Returns:
        Matplotlib axes object
    """
    gen_energy = ak.to_numpy(ak.flatten(data["reco_cand_matched_gen_energy"]))
    reco_cand_p4 = reinitialize_p4(data["reco_cand_p4s"])
    reco_energy = ak.to_numpy(ak.flatten(reco_cand_p4.energy))

    valid = gen_energy != -1
    gen_energy = gen_energy[valid]
    reco_energy = reco_energy[valid]

    if bins is None:
        en_max = max(gen_energy.max(), reco_energy.max())
        bins = np.linspace(0, en_max, 51)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    else:
        fig = ax.figure

    hist2d = ax.hist2d(
        gen_energy,
        reco_energy,
        bins=bins,
        norm=mpl.colors.LogNorm(),
        cmap="viridis",
    )

    ax.set_xlabel("Matched gen energy [GeV]")
    ax.set_ylabel("Reco candidate energy [GeV]")
    ax.set_title(f"{label} candidate energy", fontsize=12)

    cbar = plt.colorbar(hist2d[3], ax=ax)
    cbar.set_label("Number of candidates")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return ax


# =============================================================================
# Multi-dataset comparison functions
# =============================================================================


def compare_distributions(
    datasets: Dict[str, ak.Array],
    variable: str,
    bins: np.ndarray,
    xlabel: Optional[str] = None,
    ylabel: str = "Fraction [a.u.]",
    log_scale: bool = True,
    flatten: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 5.5),
) -> Figure:
    """Compare variable distributions across multiple datasets.

    Args:
        datasets: Dictionary mapping dataset name to data
        variable: Variable name to plot
        bins: Bin edges
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use log scale
        flatten: Flatten jagged arrays
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in datasets.items():
        plot_variable_distribution(
            data,
            variable,
            bins,
            label=name,
            xlabel=xlabel,
            ylabel=ylabel,
            log_scale=log_scale,
            flatten=flatten,
            ax=ax,
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def compare_jet_pt(
    datasets: Dict[str, ak.Array],
    bins: Optional[np.ndarray] = None,
    decaymode: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 5.5),
) -> Figure:
    """Compare jet pT distributions across multiple datasets.

    Args:
        datasets: Dictionary mapping dataset name to data
        bins: Bin edges
        decaymode: Optional decay mode to filter on
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if bins is None:
        bins = np.linspace(0, 220, 51)

    fig, ax = plt.subplots(figsize=figsize)

    for name, data in datasets.items():
        mask = None
        if decaymode is not None:
            mask = data["gen_jet_tau_decaymode"] == decaymode

        plot_jet_pt(data, label=name, bins=bins, decaymode_mask=mask, ax=ax)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


def compare_decay_modes(
    datasets: Dict[str, ak.Array],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5.5),
) -> Figure:
    """Compare decay mode distributions across datasets.

    Args:
        datasets: Dictionary mapping dataset name to data
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    n_datasets = len(datasets)
    fig, axs = plt.subplots(1, n_datasets, figsize=figsize)

    if n_datasets == 1:
        axes_list = [axs]
    else:
        axes_list = list(axs)

    for ax, (name, data) in zip(axes_list, datasets.items()):
        plot_decay_mode_distribution(data, ax=ax, title=f"{name} Decay modes")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return fig


# =============================================================================
# Master validation functions
# =============================================================================


def validate_ntupelizer_output(
    data: ak.Array,
    label: str = "Data",
    output_dir: Optional[str] = None,
    show_plots: bool = True,
) -> Dict[str, Figure]:
    """Run comprehensive validation plots on ntupelizer output.

    This is the master function that generates all standard validation plots
    for a single dataset from the ntupelize function.

    Args:
        data: Output from ntupelizer.ntupelize()
        label: Dataset label for legends
        output_dir: Directory to save plots (None to skip saving)
        show_plots: Whether to display plots

    Returns:
        Dictionary of figure names to figure objects
    """
    figures = {}

    # 1. Number of particles per jet
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_num_particles_per_jet(data, label=label, ax=ax)
    figures["num_particles_per_jet"] = fig
    if output_dir:
        fig.savefig(f"{output_dir}/num_particles_per_jet.pdf", bbox_inches="tight")

    # 2. Reco jet pT
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_jet_pt(data, label=label, ax=ax)
    figures["reco_jet_pt"] = fig
    if output_dir:
        fig.savefig(f"{output_dir}/reco_jet_pt.pdf", bbox_inches="tight")

    # 3. Decay mode distribution
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_decay_mode_distribution(data, ax=ax, title=f"{label} Decay modes")
    figures["decay_modes"] = fig
    if output_dir:
        fig.savefig(f"{output_dir}/decay_modes.pdf", bbox_inches="tight")

    # 4. 2D jet shapes for main decay modes
    for dm, dm_label in [(0, "h"), (10, "3h")]:
        mask = data["gen_jet_tau_decaymode"] == dm
        if ak.sum(mask) > 0:
            fig, ax = plt.subplots(figsize=(5, 5))
            plot_2d_jet_shape(
                data, decaymode_mask=mask, title=f"{label} DM{dm} ({dm_label})", ax=ax
            )
            figures[f"jet_shape_dm{dm}"] = fig
            if output_dir:
                fig.savefig(f"{output_dir}/jet_shape_dm{dm}.pdf", bbox_inches="tight")

    # 5. Lifetime variables (if present)
    lifetime_vars = ["reco_cand_dxy", "reco_cand_dz", "reco_cand_d3"]
    for var in lifetime_vars:
        if var in data.fields:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            plot_lifetime_variable(data, var, label=label, ax=ax)
            figures[var] = fig
            if output_dir:
                fig.savefig(f"{output_dir}/{var}.pdf", bbox_inches="tight")

    # 6. Gen vs reco jet energy (if fields present)
    if "gen_jet_tau_vis_energy" in data.fields:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        plot_reco_jet_energy(data, label=label, ax=ax)
        figures["reco_jet_energy"] = fig
        if output_dir:
            fig.savefig(f"{output_dir}/reco_jet_energy.pdf", bbox_inches="tight")

    # 7. Gen vs reco candidate energy (if fields present)
    if "reco_cand_matched_gen_energy" in data.fields:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        plot_reco_vs_gen_cand_energy(data, label=label, ax=ax)
        figures["reco_cand_energy"] = fig
        if output_dir:
            fig.savefig(f"{output_dir}/reco_cand_energy.pdf", bbox_inches="tight")

    if not show_plots:
        plt.close("all")

    return figures


def validate_multiple_datasets(
    datasets: Dict[str, ak.Array],
    output_dir: Optional[str] = None,
    show_plots: bool = True,
) -> Dict[str, Figure]:
    """Run comprehensive validation comparing multiple datasets.

    Args:
        datasets: Dictionary mapping dataset name to ntupelizer output
        output_dir: Directory to save plots
        show_plots: Whether to display plots

    Returns:
        Dictionary of figure names to figure objects
    """
    figures = {}

    # 1. Compare number of particles per jet
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bins = np.linspace(0, 50, 51)
    for name, data in datasets.items():
        plot_num_particles_per_jet(data, label=name, bins=bins, ax=ax)
    figures["compare_num_particles"] = fig
    if output_dir:
        fig.savefig(f"{output_dir}/compare_num_particles.pdf", bbox_inches="tight")

    # 2. Compare jet pT
    figures["compare_jet_pt"] = compare_jet_pt(
        datasets, save_path=f"{output_dir}/compare_jet_pt.pdf" if output_dir else None
    )

    # 3. Compare decay modes
    figures["compare_decay_modes"] = compare_decay_modes(
        datasets,
        save_path=f"{output_dir}/compare_decay_modes.pdf" if output_dir else None,
    )

    # 4. Compare lifetime variables
    lifetime_vars = ["reco_cand_dxy", "reco_cand_dz"]
    for var in lifetime_vars:
        if all(var in data.fields for data in datasets.values()):
            fig = compare_distributions(
                datasets,
                var,
                bins=np.linspace(0, 5, 100),
                xlabel=f"PFCandidate {var.split('_')[-1]} [mm]",
                save_path=f"{output_dir}/compare_{var}.pdf" if output_dir else None,
            )
            figures[f"compare_{var}"] = fig

    if not show_plots:
        plt.close("all")

    return figures
