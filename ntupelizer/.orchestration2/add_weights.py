#!/usr/bin/env python3
"""
Script to add correct weights to the merged ntupelized files.
Based on ntupelizer/tools/weight_tools.py and ntupelizer/scripts/set_classifier_weights.py
"""

import os
import sys
import yaml
import awkward as ak
import numpy as np
import logging
from pathlib import Path

# Add ntupelizer to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.general import reinitialize_p4

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WeightCalculator:
    """Calculates and adds weights to merged parquet files."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.all_samples_data = {}
        self.load_all_samples()

    def load_all_samples(self):
        """Load all merged samples for weight matrix calculation."""
        log.info("Loading all samples for weight calculation")
        samples_to_process = self.cfg.get("samples_to_process", [])
        all_samples = self.cfg.get("samples", {})

        for sample_name in samples_to_process:
            if sample_name not in all_samples:
                continue
            sample_cfg = all_samples[sample_name]
            output_dir = sample_cfg["output_dir"]
            merged_file = Path(output_dir) / f"{sample_name}_merged.parquet"

            if merged_file.exists():
                try:
                    log.info(f"Loading {sample_name} for weight calculation")
                    data = ak.from_parquet(merged_file)
                    self.all_samples_data[sample_name] = data
                except Exception as e:
                    log.error(f"Failed to load {sample_name}: {e}")

    def create_matrix(self, data, y_bin_edges, x_bin_edges, y_property, x_property):
        """Create histogram matrix for weight calculation."""
        if "gen_jet_p4s" not in data.fields:
            log.warning("No gen_jet_p4s field found, using basic weighting")
            return np.ones((len(y_bin_edges) - 1, len(x_bin_edges) - 1))

        p4s = reinitialize_p4(data.gen_jet_p4s)
        x_property_ = getattr(p4s, x_property).to_numpy()
        y_property_ = getattr(p4s, y_property).to_numpy()

        if y_property == "theta":
            y_property_ = np.rad2deg(y_property_)

        matrix = np.histogram2d(
            y_property_, x_property_, bins=(y_bin_edges, x_bin_edges)
        )[0]
        normalized_matrix = matrix / np.sum(matrix) if np.sum(matrix) > 0 else matrix
        return normalized_matrix

    def get_weight_matrix(self, target_matrix, comp_matrix):
        """Calculate weight matrix from target and comparison matrices."""
        weights = np.minimum(target_matrix, comp_matrix) / np.where(
            target_matrix > 0, target_matrix, 1
        )
        return np.nan_to_num(weights, nan=0.0)

    def get_event_weights(self, data, weight_matrix, theta_bin_edges, pt_bin_edges):
        """Calculate per-event weights based on kinematics."""
        if "gen_jet_p4s" not in data.fields:
            log.warning("No gen_jet_p4s field found, returning uniform weights")
            return ak.ones_like(ak.num(data, axis=0))

        p4s = reinitialize_p4(data.gen_jet_p4s)
        theta_values = np.rad2deg(p4s.theta.to_numpy())
        pt_values = p4s.pt.to_numpy()

        theta_bin = (
            np.digitize(
                theta_values, bins=(theta_bin_edges[1:] + theta_bin_edges[:-1]) / 2
            )
            - 1
        )
        pt_bin = (
            np.digitize(pt_values, bins=(pt_bin_edges[1:] + pt_bin_edges[:-1]) / 2) - 1
        )

        theta_bin = np.clip(theta_bin, 0, len(theta_bin_edges) - 2)
        pt_bin = np.clip(pt_bin, 0, len(pt_bin_edges) - 2)

        matrix_loc = np.column_stack([theta_bin, pt_bin])
        weights = ak.from_iter([weight_matrix[tuple(loc)] for loc in matrix_loc])
        return weights

    def calculate_sample_weight(self, sample_name: str, merged_data: ak.Array) -> tuple:
        """Calculate weight for a sample based on cross-section and other samples."""
        cross_sections = {
            "p8_ee_Z_Ztautau_ecm380": 1.0,
            "p8_ee_ZH_Htautau_ecm380": 0.8,
            "p8_ee_qq_ecm380": 5.0,
        }

        base_weight = cross_sections.get(sample_name, 1.0)
        n_events = len(merged_data)
        log.info(f"Sample {sample_name}: {n_events} events, base weight: {base_weight}")

        if len(self.all_samples_data) > 1:
            eta_bin_edges = np.linspace(-3, 3, 10)
            pt_bin_edges = np.linspace(0, 200, 20)

            sample_matrix = self.create_matrix(
                merged_data,
                eta_bin_edges,
                pt_bin_edges,
                y_property="eta",
                x_property="pt",
            )

            reference_sample = None
            for ref_name, ref_data in self.all_samples_data.items():
                if ref_name != sample_name:
                    reference_sample = ref_data
                    break

            if reference_sample is not None:
                ref_matrix = self.create_matrix(
                    reference_sample,
                    eta_bin_edges,
                    pt_bin_edges,
                    y_property="eta",
                    x_property="pt",
                )
                weight_matrix = self.get_weight_matrix(
                    target_matrix=ref_matrix, comp_matrix=sample_matrix
                )
                kinematic_weights = self.get_event_weights(
                    merged_data, weight_matrix, eta_bin_edges, pt_bin_edges
                )
                return base_weight, kinematic_weights

        uniform_weights = ak.ones_like(ak.num(merged_data, axis=0))
        return base_weight, uniform_weights

    def add_weights_to_sample(self, sample_name: str, output_dir: str, is_signal: bool):
        """Add weights to merged file for a given sample."""
        log.info(f"Adding weights for sample: {sample_name}")

        merged_file = Path(output_dir) / f"{sample_name}_merged.parquet"
        if not merged_file.exists():
            log.error(f"Merged file not found: {merged_file}")
            return

        weighted_file = Path(output_dir) / f"{sample_name}_weighted.parquet"
        if weighted_file.exists():
            log.info(f"Weighted file already exists: {weighted_file}")
            return

        log.info(f"Reading merged file: {merged_file}")
        data = ak.from_parquet(merged_file)
        log.info(f"Data length: {len(data)}")
        log.info(f"Data fields: {list(data.fields)}")

        sample_weight, kinematic_weights = self.calculate_sample_weight(
            sample_name, data
        )

        weighted_data = {field: data[field] for field in data.fields}
        weighted_data["sample_weight"] = ak.full_like(
            ak.num(data, axis=0), sample_weight
        )
        weighted_data["kinematic_weight"] = kinematic_weights
        weighted_data["is_signal"] = ak.full_like(ak.num(data, axis=0), is_signal)
        weighted_data["sample_name"] = ak.full_like(
            ak.num(data, axis=0), sample_name, dtype="<U50"
        )
        weighted_data["final_weight"] = (
            weighted_data["sample_weight"] * weighted_data["kinematic_weight"]
        )

        weighted_array = ak.Array(weighted_data)

        log.info(f"Saving weighted file: {weighted_file}")
        ak.to_parquet(weighted_array, weighted_file, row_group_size=1024)

        log.info(f"Added weights to {len(data)} events")
        log.info(f"Sample weight: {sample_weight}")
        log.info(
            f"Kinematic weights range: {ak.min(kinematic_weights):.3f} - {ak.max(kinematic_weights):.3f}"
        )
        log.info(f"Is signal: {is_signal}")

    def run(self):
        """Add weights to all configured samples."""
        samples_to_process = self.cfg.get("samples_to_process", [])
        all_samples = self.cfg.get("samples", {})

        if not samples_to_process:
            log.warning("No samples specified in samples_to_process")
            return

        log.info(
            f"Will add weights for {len(samples_to_process)} samples: {samples_to_process}"
        )

        for sample_name in samples_to_process:
            if sample_name not in all_samples:
                log.error(f"Sample {sample_name} not found in samples configuration")
                continue

            sample_cfg = all_samples[sample_name]
            output_dir = sample_cfg["output_dir"]
            is_signal = sample_cfg.get("is_signal", False)

            if not os.path.exists(output_dir):
                log.warning(f"Output directory does not exist: {output_dir}")
                continue

            self.add_weights_to_sample(sample_name, output_dir, is_signal)

        log.info("Weight assignment completed")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: add_weights.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    calculator = WeightCalculator(config_path)
    calculator.run()


if __name__ == "__main__":
    main()
