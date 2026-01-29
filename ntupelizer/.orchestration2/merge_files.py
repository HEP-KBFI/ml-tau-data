#!/usr/bin/env python3
"""
Script to merge all ntupelized parquet files for each sample.
Based on ntupelizer/scripts/mergeFiles.py
"""

import os
import sys
import glob
import yaml
import awkward as ak
import numpy as np
import logging
from pathlib import Path
from typing import List
import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FileMerger:
    """Merges ntupelized parquet files for each sample."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get_parquet_files(self, output_dir: str) -> List[str]:
        """Get all parquet files in the output directory."""
        pattern = str(Path(output_dir) / "*.parquet")
        files = glob.glob(pattern)

        # Exclude already merged files
        files = [f for f in files if not f.endswith("_merged.parquet")]

        files.sort()
        return files

    def load_sample_data(self, output_dir: str, sample_name: str) -> ak.Array:
        """Load and merge all parquet files for a sample using awkward array."""
        log.info(f"Loading sample data for: {sample_name}")

        # Get all parquet files
        parquet_files = self.get_parquet_files(output_dir)

        if not parquet_files:
            log.warning(f"No parquet files found in {output_dir}")
            return None

        log.info(f"Found {len(parquet_files)} files to merge")

        # Load all files using awkward
        data_arrays = []
        for file_path in tqdm.tqdm(parquet_files, desc="Loading files"):
            try:
                arr = ak.from_parquet(file_path)
                # Ensure it's a proper awkward array
                arr = ak.Array({k: arr[k] for k in arr.fields})
                data_arrays.append(arr)
            except Exception as e:
                log.error(f"Failed to read {file_path}: {e}")
                continue

        if not data_arrays:
            log.error(f"No files could be read for sample {sample_name}")
            return None

        # Concatenate all arrays
        log.info("Concatenating arrays...")
        merged_data = ak.concatenate(data_arrays)

        # Shuffle data as in original mergeFiles.py
        log.info("Shuffling data...")
        perm = np.random.permutation(len(merged_data))
        merged_data = merged_data[perm]

        log.info(f"Merged array length: {len(merged_data)}")
        return merged_data

    def merge_sample_files(self, sample_name: str, output_dir: str):
        """Merge all parquet files for a given sample."""
        log.info(f"Merging files for sample: {sample_name}")

        # Check if merged file already exists
        merged_file = Path(output_dir) / f"{sample_name}_merged.parquet"
        if merged_file.exists():
            log.info(f"Merged file already exists: {merged_file}")
            return

        # Load and merge data
        merged_data = self.load_sample_data(output_dir, sample_name)
        if merged_data is None:
            return

        # Save merged file
        log.info(f"Saving merged file: {merged_file}")
        ak.to_parquet(merged_data, merged_file, row_group_size=1024)

        log.info(f"Successfully merged {len(merged_data)} events into {merged_file}")

    def run(self):
        """Merge files for all configured samples."""
        samples_to_process = self.cfg.get("samples_to_process", [])
        all_samples = self.cfg.get("samples", {})

        if not samples_to_process:
            log.warning("No samples specified in samples_to_process")
            return

        log.info(
            f"Will merge files for {len(samples_to_process)} samples: {samples_to_process}"
        )

        for sample_name in samples_to_process:
            if sample_name not in all_samples:
                log.error(f"Sample {sample_name} not found in samples configuration")
                continue

            sample_cfg = all_samples[sample_name]
            output_dir = sample_cfg["output_dir"]

            if not os.path.exists(output_dir):
                log.warning(f"Output directory does not exist: {output_dir}")
                continue

            self.merge_sample_files(sample_name, output_dir)

        log.info("Merging completed")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: merge_files.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    merger = FileMerger(config_path)
    merger.run()


if __name__ == "__main__":
    main()
