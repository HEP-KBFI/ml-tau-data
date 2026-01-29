#!/usr/bin/env python3
"""
Script to read configuration and orchestrate ntupelization of all files.
Submits SLURM jobs for each batch of files to be processed.
"""

import os
import sys
import glob
import subprocess
import hydra
import logging
from pathlib import Path
from typing import List, Dict, Any
import math
from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NtupelizerOrchestrator:
    """Orchestrates the ntupelization of all configured samples."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.orchestration_dir = Path(__file__).parent
        self.batch_script = self.orchestration_dir / "ntupelize_batch.sh"
        self.use_slurm = self.cfg.get("slurm_run", False)
        self.files_per_job = self.cfg.get("files_per_job", 20)
        self.n_files = self.cfg.get("n_files", -1)
        self.n_files_per_sample = self.cfg.get("n_files_per_sample", -1)

        self.job_dir = Path.home() / "tmp" / "ntupelizer_jobs"
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(self.orchestration_dir / "templates")
        )

    def get_input_files(self, sample_name: str, input_dir: str) -> List[str]:
        """Get list of input files for a sample."""
        # Check if paths override is provided
        if hasattr(self.cfg, "paths") and self.cfg.paths:
            custom_paths = self.cfg.paths
            if isinstance(custom_paths, str):
                custom_paths = [custom_paths]

            files = []
            for path_pattern in custom_paths:
                if "*" in path_pattern:
                    files.extend(glob.glob(path_pattern))
                else:
                    if os.path.isfile(path_pattern):
                        files.append(path_pattern)
                    elif os.path.isdir(path_pattern):
                        files.extend(glob.glob(os.path.join(path_pattern, "*.root")))

            log.info(f"Using custom paths: {custom_paths}")
        else:
            input_path = Path(input_dir)
            pattern = str(input_path / "*.root")
            files = glob.glob(pattern)

            if not files:
                pattern = str(input_path / "**" / "*.root")
                files = glob.glob(pattern, recursive=True)

        if not files:
            log.warning(f"No .root files found")
            return []

        files.sort()

        if self.n_files > 0:
            files = files[: self.n_files]

        if self.n_files_per_sample > 0:
            files = files[: self.n_files_per_sample]

        log.info(f"Found {len(files)} files for sample {sample_name}")
        return files

    def chunk_files(self, files: List[str]) -> List[List[str]]:
        """Split files into chunks for processing."""
        chunks = []
        for i in range(0, len(files), self.files_per_job):
            chunk = files[i : i + self.files_per_job]
            chunks.append(chunk)
        return chunks

    def submit_slurm_job(
        self,
        sample_name: str,
        chunk_idx: int,
        input_files: List[str],
        output_dir: str,
        is_signal: bool,
    ) -> str:
        """Submit a SLURM job for processing a chunk of files."""
        job_name = f"ntup_{sample_name}_{chunk_idx}"

        # Create job script
        job_script_path = self.job_dir / f"{job_name}.sh"

        slurm_cfg = self.cfg.get("slurm", {})
        partition = slurm_cfg.get("partition", "main")
        walltime = slurm_cfg.get("walltime", "04:00:00")
        memory = slurm_cfg.get("memory", 8000)
        cpus = slurm_cfg.get("cpus_per_task", 1)
        nodes = slurm_cfg.get("nodes", 1)
        ntasks = slurm_cfg.get("ntasks", 1)

        file_list = " ".join(input_files)

        template = self.jinja_env.get_template("slurm_job.sh.j2")
        job_script_content = template.render(
            job_name=job_name,
            job_dir=self.job_dir,
            partition=partition,
            walltime=walltime,
            memory=memory,
            cpus=cpus,
            nodes=nodes,
            ntasks=ntasks,
            working_dir=self.orchestration_dir.parent.parent,
            batch_script=self.batch_script,
            sample_name=sample_name,
            output_dir=output_dir,
            is_signal=is_signal,
            file_list=file_list,
        )

        with open(job_script_path, "w") as f:
            f.write(job_script_content)

        # Submit job
        cmd = ["sbatch", str(job_script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]

        log.info(f"Submitted job {job_id} for {sample_name} chunk {chunk_idx}")
        return job_id

    def run_local_job(
        self,
        sample_name: str,
        chunk_idx: int,
        input_files: List[str],
        output_dir: str,
        is_signal: bool,
    ):
        """Run processing locally for a chunk of files."""
        log.info(f"Processing {sample_name} chunk {chunk_idx} locally")

        file_list = " ".join(input_files)

        cmd = [
            str(self.batch_script),
            sample_name,
            output_dir,
            str(is_signal).lower(),
            file_list,
        ]

        try:
            subprocess.run(cmd, check=True, cwd=self.orchestration_dir.parent.parent)
            log.info(f"Completed {sample_name} chunk {chunk_idx}")
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to process {sample_name} chunk {chunk_idx}: {e}")
            raise

    def process_sample(self, sample_name: str, sample_cfg: Dict[str, Any]):
        """Process all files for a given sample."""
        input_dir = sample_cfg["input_dir"]
        output_dir = sample_cfg["output_dir"]
        is_signal = sample_cfg.get("is_signal", False)

        log.info(f"Processing sample: {sample_name}")
        log.info(f"  Input dir: {input_dir}")
        log.info(f"  Output dir: {output_dir}")
        log.info(f"  Is signal: {is_signal}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get input files
        input_files = self.get_input_files(sample_name, input_dir)
        if not input_files:
            log.warning(f"No files found for sample {sample_name}, skipping")
            return

        # Chunk files
        chunks = self.chunk_files(input_files)
        log.info(
            f"Split into {len(chunks)} chunks of up to {self.files_per_job} files each"
        )

        # Process chunks
        job_ids = []
        for chunk_idx, chunk_files in enumerate(chunks):
            if self.use_slurm:
                job_id = self.submit_slurm_job(
                    sample_name, chunk_idx, chunk_files, output_dir, is_signal
                )
                job_ids.append(job_id)
            else:
                self.run_local_job(
                    sample_name, chunk_idx, chunk_files, output_dir, is_signal
                )

        if job_ids:
            log.info(
                f"Submitted {len(job_ids)} jobs for {sample_name}: {', '.join(job_ids)}"
            )

    def run(self):
        """Run ntupelization for all configured samples."""
        samples_to_process = self.cfg.get("samples_to_process", [])
        all_samples = self.cfg.get("samples", {})

        if not samples_to_process:
            log.warning("No samples specified in samples_to_process")
            return

        log.info(
            f"Will process {len(samples_to_process)} samples: {samples_to_process}"
        )

        for sample_name in samples_to_process:
            if sample_name not in all_samples:
                log.error(f"Sample {sample_name} not found in samples configuration")
                continue

            self.process_sample(sample_name, all_samples[sample_name])

        if self.use_slurm:
            log.info("All SLURM jobs submitted. Monitor with 'squeue -u $USER'")
        else:
            log.info("All local processing completed")


@hydra.main(version_base=None, config_path="../config", config_name="ntupelizer")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    orchestrator = NtupelizerOrchestrator(cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
