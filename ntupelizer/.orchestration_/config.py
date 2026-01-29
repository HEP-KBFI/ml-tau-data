"""
Configuration utilities for law workflow.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from omegaconf import OmegaConf, DictConfig, ListConfig


@dataclass
class SampleConfig:
    """Configuration for a single sample."""

    name: str
    input_dir: str
    output_dir: str
    is_signal: bool = True

    @property
    def input_files(self) -> List[str]:
        """Get list of input ROOT files."""
        pattern = os.path.join(self.input_dir, "**", "*.root")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            # Try non-recursive pattern
            pattern = os.path.join(self.input_dir, "*.root")
            files = sorted(glob.glob(pattern))
        return files

    def get_output_path(self, input_file: str) -> str:
        """Get output parquet path for a given input file."""
        basename = os.path.basename(input_file).replace(".root", ".parquet")
        return os.path.join(self.output_dir, basename)


@dataclass
class SlurmConfig:
    """SLURM configuration for workflow jobs."""

    partition: str = "main"
    walltime: str = "04:00:00"
    memory: int = 8000  # MB
    cpus_per_task: int = 1


@dataclass
class WorkflowConfig:
    """Configuration for the entire workflow."""

    config_path: str
    cfg: Union[DictConfig, ListConfig] = field(init=False, repr=False)
    samples: Dict[str, SampleConfig] = field(default_factory=dict)
    samples_to_process: List[str] = field(default_factory=list)
    files_per_job: int = 20
    n_files: int = -1
    validation_output_dir: Optional[str] = None
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __post_init__(self):
        """Load configuration from YAML file."""
        self.cfg = OmegaConf.load(self.config_path)

        # Parse samples
        for name, sample_cfg in self.cfg.samples.items():
            self.samples[name] = SampleConfig(
                name=name,
                input_dir=sample_cfg.input_dir,
                output_dir=sample_cfg.output_dir,
                is_signal=OmegaConf.select(sample_cfg, "is_signal", default=True),
            )

        # Get samples to process
        self.samples_to_process = list(
            OmegaConf.select(
                self.cfg, "samples_to_process", default=list(self.samples.keys())
            )
        )
        self.files_per_job = OmegaConf.select(self.cfg, "files_per_job", default=20)
        self.n_files = OmegaConf.select(self.cfg, "n_files", default=-1)

        # Validation output
        if "validation" in self.cfg:
            val_cfg = self.cfg.validation
            self.validation_output_dir = os.path.expandvars(
                OmegaConf.select(
                    val_cfg, "output_dir", default="$HOME/ntuple_validation"
                )
            )

        # SLURM config
        if "slurm" in self.cfg:
            slurm_cfg = self.cfg.slurm
            self.slurm = SlurmConfig(
                partition=OmegaConf.select(slurm_cfg, "partition", default="main"),
                walltime=OmegaConf.select(slurm_cfg, "walltime", default="04:00:00"),
                memory=OmegaConf.select(slurm_cfg, "memory", default=8000),
                cpus_per_task=OmegaConf.select(slurm_cfg, "cpus_per_task", default=1),
            )

    def get_sample(self, name: str) -> SampleConfig:
        """Get sample configuration by name."""
        if name not in self.samples:
            raise ValueError(f"Sample '{name}' not found in configuration")
        return self.samples[name]

    def get_samples_to_process(self) -> List[SampleConfig]:
        """Get list of samples to process."""
        return [self.samples[name] for name in self.samples_to_process]

    def get_input_chunks(self, sample_name: str) -> List[List[str]]:
        """
        Get input file chunks for a sample.

        Each chunk will be processed together and produce one output file.
        """
        import numpy as np

        sample = self.get_sample(sample_name)
        input_files = sample.input_files

        # Apply n_files limit
        if self.n_files > 0:
            input_files = input_files[: self.n_files]

        if len(input_files) == 0:
            return []

        # Split into chunks
        n_chunks = max(1, len(input_files) // self.files_per_job)
        chunks = np.array_split(input_files, n_chunks)
        return [list(chunk) for chunk in chunks if len(chunk) > 0]


def load_config(config_path: str) -> WorkflowConfig:
    """Load workflow configuration from YAML file."""
    return WorkflowConfig(config_path=config_path)


# Config type mapping: "old" -> ntupelizer.yaml, "new" -> podio_root_ntupelizer.yaml
CONFIG_TYPE_MAP = {
    "old": "ntupelizer.yaml",
    "new": "podio_root_ntupelizer.yaml",
}


def get_config_path(config_type: str = "new") -> str:
    """
    Get configuration file path based on config type.

    Args:
        config_type: Either "old" (ntupelizer.yaml) or "new" (podio_root_ntupelizer.yaml)

    Returns:
        Absolute path to the configuration file
    """
    if config_type not in CONFIG_TYPE_MAP:
        raise ValueError(
            f"Invalid config_type '{config_type}'. Must be one of: {list(CONFIG_TYPE_MAP.keys())}"
        )

    package_dir = Path(__file__).parent.parent
    return str(package_dir / "config" / CONFIG_TYPE_MAP[config_type])


def get_default_config_path() -> str:
    """Get default configuration file path (new/podio_root_ntupelizer)."""
    return get_config_path("new")
