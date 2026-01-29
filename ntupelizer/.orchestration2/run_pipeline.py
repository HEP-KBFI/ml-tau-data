#!/usr/bin/env python3
"""
Main pipeline runner for ntupelization workflow.
Orchestrates the full process: ntupelize -> merge -> weights -> validate
"""

import os
import sys
import subprocess
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

# Add ntupelizer to path
sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


class PipelineRunner:
    """Main pipeline orchestrator."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.orchestration_dir = Path(__file__).parent
        self.base_dir = self.orchestration_dir.parent.parent

    def run_step(self, step_name: str, script_path: str, *args) -> bool:
        """Run a pipeline step and return success status."""
        log.info(f"Starting step: {step_name}")

        cmd = [str(script_path)] + list(args)
        log.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            log.info(f"Step {step_name} completed successfully")
            if result.stdout:
                log.info(f"Output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"Step {step_name} failed with return code {e.returncode}")
            log.error(f"Error output: {e.stderr}")
            return False

    def run_pipeline(self):
        """Run the complete pipeline."""
        log.info("Starting ntupelization pipeline")

        # Step 1: Ntupelize files
        ntupelize_script = self.orchestration_dir / "ntupelize_files.py"
        if not self.run_step(
            "Ntupelize", ntupelize_script, str(self.orchestration_dir / "config.yaml")
        ):
            log.error("Ntupelization failed, stopping pipeline")
            return False

        # Step 2: Merge files
        merge_script = self.orchestration_dir / "merge_files.py"
        if not self.run_step(
            "Merge", merge_script, str(self.orchestration_dir / "config.yaml")
        ):
            log.error("Merge failed, stopping pipeline")
            return False

        # Step 3: Add weights
        weights_script = self.orchestration_dir / "add_weights.py"
        if not self.run_step(
            "Add weights", weights_script, str(self.orchestration_dir / "config.yaml")
        ):
            log.error("Weight assignment failed, stopping pipeline")
            return False

        # Step 4: Validation
        validation_script = self.orchestration_dir / "validate.py"
        if not self.run_step(
            "Validate", validation_script, str(self.orchestration_dir / "config.yaml")
        ):
            log.error("Validation failed, stopping pipeline")
            return False

        log.info("Pipeline completed successfully!")
        return True


@hydra.main(version_base=None, config_path="../config", config_name="ntupelizer")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Save config to orchestration directory for other scripts
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "w") as f:
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, f)

    runner = PipelineRunner(cfg)
    success = runner.run_pipeline()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
