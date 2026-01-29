#!/usr/bin/env python3
"""
Script to create validation plots from the weighted ntupelized files.
"""

import os
import sys
import yaml
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import vector
import logging
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ValidationPlotter:
    """Creates validation plots from weighted ntupelized data."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Set up matplotlib for non-interactive backend
        plt.style.use("default")
        sns.set_palette("husl")

        # Create validation output directory
        validation_cfg = self.cfg.get("validation", {})
        validation_dir = validation_cfg.get("output_dir", "$HOME/ntuple_validation")
        self.validation_dir = Path(os.path.expandvars(validation_dir))
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Validation output directory: {self.validation_dir}")

    def load_weighted_data(self, sample_name: str, output_dir: str) -> ak.Array:
        """Load weighted data for a sample."""
        weighted_file = Path(output_dir) / f"{sample_name}_weighted.parquet"

        if not weighted_file.exists():
            log.error(f"Weighted file not found: {weighted_file}")
            return None

        log.info(f"Loading weighted data: {weighted_file}")
        data = ak.from_parquet(weighted_file)
        log.info(f"Loaded {len(data)} events for {sample_name}")

        return data

    def plot_basic_distributions(self, data_dict: Dict[str, ak.Array]):
        """Create basic distribution plots."""
        log.info("Creating basic distribution plots")

        # Define key variables to plot based on typical ntupelizer output
        plot_vars = [
            ("gen_jet_p4s.pt", "Jet pT [GeV]", (0, 200)),
            ("gen_jet_p4s.eta", "Jet η", (-3, 3)),
            ("gen_jet_p4s.phi", "Jet φ", (-np.pi, np.pi)),
            ("gen_jet_p4s.mass", "Jet mass [GeV]", (0, 10)),
        ]

        for var_name, xlabel, xlim in plot_vars:
            fig, ax = plt.subplots(figsize=(10, 6))

            for sample_name, data in data_dict.items():
                if var_name in data.fields or "." in var_name:
                    try:
                        # Handle nested field access
                        if "." in var_name:
                            field_parts = var_name.split(".")
                            var_data = data[field_parts[0]]
                            for part in field_parts[1:]:
                                var_data = getattr(var_data, part)
                        else:
                            var_data = data[var_name]

                        # Convert to numpy for plotting
                        var_values = ak.to_numpy(ak.flatten(var_data))

                        # Apply weights if available
                        weights = None
                        if "final_weight" in data.fields:
                            weights = ak.to_numpy(ak.flatten(data["final_weight"]))
                            if len(weights) != len(var_values):
                                weights = np.ones_like(var_values)

                        ax.hist(
                            var_values,
                            bins=50,
                            alpha=0.7,
                            label=sample_name,
                            weights=weights,
                            density=True,
                        )
                    except Exception as e:
                        log.warning(f"Could not plot {var_name} for {sample_name}: {e}")

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Normalized entries")
            ax.set_xlim(xlim)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = self.validation_dir / f"dist_{var_name.replace('.', '_')}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved plot: {output_path}")
            plt.close()

    def plot_correlation_matrix(self, data_dict: Dict[str, pd.DataFrame]):
        """Create correlation matrix plots for each sample."""
        log.info("Creating correlation matrices")

        # Variables for correlation analysis
        corr_vars = [
            "tau_pt",
            "tau_eta",
            "tau_phi",
            "tau_mass",
            "tau_energy",
            "n_tracks",
            "n_clusters",
        ]

        for sample_name, df in data_dict.items():
            # Select variables that exist in the dataframe
            available_vars = [var for var in corr_vars if var in df.columns]

            if len(available_vars) < 2:
                log.warning(
                    f"Not enough variables for correlation plot in {sample_name}"
                )
                continue

            # Calculate correlation matrix
            corr_df = df[available_vars].corr()

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_df, annot=True, cmap="coolwarm", center=0, square=True, ax=ax
            )
            ax.set_title(f"Variable Correlations - {sample_name}")

            plt.tight_layout()
            output_path = self.validation_dir / f"correlation_{sample_name}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved correlation plot: {output_path}")
            plt.close()

    def plot_event_counts(self, data_dict: Dict[str, ak.Array]):
        """Create event count comparison plots."""
        log.info("Creating event count plots")

        sample_names = []
        raw_counts = []
        weighted_counts = []

        for sample_name, data in data_dict.items():
            sample_names.append(sample_name)
            raw_counts.append(len(data))

            if "final_weight" in data.fields:
                weighted_counts.append(float(ak.sum(data["final_weight"])))
            else:
                weighted_counts.append(len(data))

        # Create comparison plot
        x = np.arange(len(sample_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width / 2, raw_counts, width, label="Raw events", alpha=0.7)
        bars2 = ax.bar(
            x + width / 2, weighted_counts, width, label="Weighted events", alpha=0.7
        )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Event count")
        ax.set_title("Event Counts by Sample")
        ax.set_xticks(x)
        ax.set_xticklabels(sample_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1e}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        output_path = self.validation_dir / "event_counts.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved event count plot: {output_path}")
        plt.close()

    def create_summary_report(self, data_dict: Dict[str, ak.Array]):
        """Create a summary report of the validation."""
        log.info("Creating summary report")

        report_lines = []
        report_lines.append("# Ntupelization Validation Report\n")

        # Sample summary
        report_lines.append("## Sample Summary\n")
        for sample_name, data in data_dict.items():
            report_lines.append(f"### {sample_name}")
            report_lines.append(f"- Raw events: {len(data):,}")

            if "final_weight" in data.fields:
                total_weight = float(ak.sum(data["final_weight"]))
                report_lines.append(f"- Weighted events: {total_weight:.2e}")
                report_lines.append(f"- Average weight: {total_weight / len(data):.2e}")

            if "is_signal" in data.fields:
                is_signal = bool(ak.any(data["is_signal"])) if len(data) > 0 else False
                report_lines.append(f"- Signal sample: {is_signal}")

            report_lines.append(f"- Variables: {len(data.fields)}")
            report_lines.append("")

        # Variable summary
        report_lines.append("## Variable Summary\n")
        all_vars = set()
        for data in data_dict.values():
            all_vars.update(data.fields)

        report_lines.append(f"Total unique variables: {len(all_vars)}")
        report_lines.append(f"Variables: {', '.join(sorted(all_vars))}")
        report_lines.append("")

        # Write report
        report_path = self.validation_dir / "validation_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        log.info(f"Saved validation report: {report_path}")

    def run(self):
        """Run validation for all configured samples."""
        validation_samples = self.cfg.get("validation", {}).get(
            "validation_samples", []
        )
        all_samples = self.cfg.get("samples", {})

        if not validation_samples:
            log.warning("No validation samples specified")
            return

        log.info(
            f"Will validate {len(validation_samples)} samples: {validation_samples}"
        )

        # Load data for all samples
        data_dict = {}

        for sample_name in validation_samples:
            if sample_name not in all_samples:
                log.error(f"Sample {sample_name} not found in samples configuration")
                continue

            sample_cfg = all_samples[sample_name]
            output_dir = sample_cfg["output_dir"]

            df = self.load_weighted_data(sample_name, output_dir)
            if df is not None:
                data_dict[sample_name] = df

        if not data_dict:
            log.error("No data loaded for validation")
            return

        # Create plots
        self.plot_basic_distributions(data_dict)
        self.plot_correlation_matrix(data_dict)
        self.plot_event_counts(data_dict)
        self.create_summary_report(data_dict)

        log.info(f"Validation completed. Results saved to: {self.validation_dir}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: validate.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    plotter = ValidationPlotter(config_path)
    plotter.run()


if __name__ == "__main__":
    main()
