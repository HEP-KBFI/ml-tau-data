"""
Law tasks for ntupelizer workflow.

This module provides Luigi Analysis Workflow (law) tasks for:
- NtupelizeFile: Process a single ROOT file
- NtupelizeSample: Process all files in a sample (supports SLURM)
- NtupelizeAllSamples: Process all configured samples
- ValidateSample: Validate a single sample's output
- ValidateAllSamples: Validate all samples
- FullPipeline: Run ntupelization + validation
"""

import os
import glob

import law
import luigi
import awkward as ak

# Load contrib modules
law.contrib.load("slurm")

from law.contrib.slurm import SlurmWorkflow

from ntupelizer.orchestration_.config import (
    WorkflowConfig,
    load_config,
    get_config_path,
)


# =============================================================================
# Base task with common configuration
# =============================================================================


class NtupelizeBaseTask(law.Task):
    """Base task with common configuration parameters."""

    config_type = luigi.Parameter(
        default="new",
        description="Config type: 'old' (ntupelizer.yaml) or 'new' (podio_root_ntupelizer.yaml)",
    )

    job_workflow = luigi.Parameter(
        default="slurm",
        description="Workflow for jobs: 'slurm' (submit to cluster) or 'local' (run locally)",
    )

    use_container = luigi.BoolParameter(
        default=True,
        description="Run SLURM jobs inside container using ./run.sh (default: True)",
    )

    @property
    def config_path(self) -> str:
        """Get the config path based on config_type."""
        return get_config_path(self.config_type)

    @property
    def workflow_config(self) -> WorkflowConfig:
        """Load and cache workflow configuration."""
        if not hasattr(self, "_workflow_config"):
            self._workflow_config = load_config(self.config_path)
        return self._workflow_config

    @property
    def cfg(self):
        """Get the raw OmegaConf configuration."""
        return self.workflow_config.cfg


# =============================================================================
# File-level ntupelization task (runs on SLURM worker)
# =============================================================================


class NtupelizeFile(NtupelizeBaseTask):
    """
    Ntupelize a single ROOT file or chunk of files.

    Parameters:
        sample_name: Name of the sample (must be in config)
        chunk_index: Index of the file chunk to process
        config_type: Config type ('old' or 'new')
    """

    sample_name = luigi.Parameter(description="Name of the sample to process")
    chunk_index = luigi.IntParameter(
        default=0, description="Index of the file chunk to process"
    )

    def output(self):
        """Define output parquet file target."""
        sample = self.workflow_config.get_sample(self.sample_name)
        chunks = self.workflow_config.get_input_chunks(self.sample_name)

        if self.chunk_index >= len(chunks):
            raise ValueError(
                f"Chunk index {self.chunk_index} out of range for sample {self.sample_name}"
            )

        # Output path based on first file in chunk
        first_file = chunks[self.chunk_index][0]
        output_path = sample.get_output_path(first_file)

        return law.LocalFileTarget(output_path)

    def run(self):
        """Run ntupelization on the file chunk."""
        from ntupelizer.tools.ntupelizer import PodioROOTNtuplelizer

        sample = self.workflow_config.get_sample(self.sample_name)
        chunks = self.workflow_config.get_input_chunks(self.sample_name)
        input_files = chunks[self.chunk_index]

        print(f"Processing chunk {self.chunk_index} with {len(input_files)} files")
        print(f"Input files: {input_files[:3]}{'...' if len(input_files) > 3 else ''}")

        # Initialize ntupelizer
        ntupelizer = PodioROOTNtuplelizer(cfg=self.cfg)

        # Process each file and collect results
        all_data = []
        for input_path in input_files:
            print(f"Processing: {input_path}")
            try:
                data = ntupelizer.ntupelize(
                    input_path=input_path,
                    signal_sample=sample.is_signal,
                    remove_background=sample.is_signal,
                )
                # Convert Record to Array for concatenation
                data = ak.Array({k: data[k] for k in data.fields})
                all_data.append(data)
            except (OSError, ValueError, RuntimeError) as e:
                print(f"Error processing {input_path}: {e}")
                continue

        if not all_data:
            raise RuntimeError(f"No data processed for chunk {self.chunk_index}")

        # Merge all data
        merged_data = ak.concatenate(all_data)
        print(f"Total jets after merging: {len(merged_data)}")

        # Save output
        output_path = self.output().path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ak.to_parquet(merged_data, output_path)
        print(f"Saved to {output_path}")


# =============================================================================
# Sample-level ntupelization task (SLURM workflow)
# =============================================================================


class NtupelizeSample(NtupelizeBaseTask, SlurmWorkflow, law.LocalWorkflow):
    """
    Ntupelize all files in a sample using SLURM.

    This is a workflow task that spawns NtupelizeFile tasks for each chunk,
    submitting them as SLURM jobs.

    Parameters:
        sample_name: Name of the sample to process
        config_type: Config type ('old' or 'new')

    SLURM Parameters (can be overridden via CLI):
        --slurm-partition: SLURM partition to use
        --slurm-walltime: Maximum walltime (e.g., '04:00:00')
        --slurm-memory: Memory per job (e.g., '8000' for 8GB)
    """

    sample_name = luigi.Parameter(description="Name of the sample to process")

    # SLURM configuration (CLI overrides config file defaults)
    slurm_partition = luigi.Parameter(
        default="",
        description="SLURM partition (default: from config)",
    )
    slurm_walltime = luigi.Parameter(
        default="",
        description="Maximum walltime (default: from config)",
    )
    slurm_memory = luigi.Parameter(
        default="",
        description="Memory per job in MB (default: from config)",
    )

    def _get_slurm_partition(self):
        """Get SLURM partition from CLI or config."""
        return self.slurm_partition or self.workflow_config.slurm.partition

    def _get_slurm_walltime(self):
        """Get SLURM walltime from CLI or config."""
        return self.slurm_walltime or self.workflow_config.slurm.walltime

    def _get_slurm_memory(self):
        """Get SLURM memory from CLI or config."""
        return self.slurm_memory or str(self.workflow_config.slurm.memory)

    def create_branch_map(self):
        """Create a mapping of branch index to chunk index."""
        chunks = self.workflow_config.get_input_chunks(self.sample_name)
        return {i: i for i in range(len(chunks))}

    def output(self):
        """Define output for each branch."""
        sample = self.workflow_config.get_sample(self.sample_name)
        chunks = self.workflow_config.get_input_chunks(self.sample_name)

        if self.branch >= len(chunks):
            raise ValueError(
                f"Branch {self.branch} out of range for sample {self.sample_name}"
            )

        first_file = chunks[self.branch][0]
        output_path = sample.get_output_path(first_file)
        return law.LocalFileTarget(output_path)

    def run(self):
        """Run ntupelization on this branch's file chunk."""
        from ntupelizer.tools.ntupelizer import PodioROOTNtuplelizer

        sample = self.workflow_config.get_sample(self.sample_name)
        chunks = self.workflow_config.get_input_chunks(self.sample_name)
        input_files = chunks[self.branch]

        print(f"Processing branch {self.branch} with {len(input_files)} files")
        print(f"Input files: {input_files[:3]}{'...' if len(input_files) > 3 else ''}")

        # Initialize ntupelizer
        ntupelizer = PodioROOTNtuplelizer(cfg=self.cfg)

        # Process each file and collect results
        all_data = []
        for input_path in input_files:
            print(f"Processing: {input_path}")
            try:
                data = ntupelizer.ntupelize(
                    input_path=input_path,
                    signal_sample=sample.is_signal,
                    remove_background=sample.is_signal,
                )
                # Convert Record to Array for concatenation
                data = ak.Array({k: data[k] for k in data.fields})
                all_data.append(data)
            except (OSError, ValueError, RuntimeError) as e:
                print(f"Error processing {input_path}: {e}")
                continue

        if not all_data:
            raise RuntimeError(f"No data processed for branch {self.branch}")

        # Merge all data
        merged_data = ak.concatenate(all_data)
        print(f"Total jets after merging: {len(merged_data)}")

        # Save output
        output_path = self.output().path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ak.to_parquet(merged_data, output_path)
        print(f"Saved to {output_path}")

    def slurm_resources(self, job_num, branches):
        """Define SLURM resources for the job."""
        return {
            "partition": self._get_slurm_partition(),
            "walltime": self._get_slurm_walltime(),
            "memory": self._get_slurm_memory(),
        }

    def slurm_output_directory(self):
        """Define directory for SLURM job logs."""
        sample = self.workflow_config.get_sample(self.sample_name)
        log_dir = os.path.join(sample.output_dir, "slurm_logs")
        return law.LocalDirectoryTarget(log_dir)

    def slurm_job_file_content(self):
        """Override job script to run law commands inside container."""
        import os

        content = super().slurm_job_file_content()

        # Replace law commands to run inside container where environment works
        # This avoids the venv symlink issue completely
        content = content.replace(" law ", " ./run.sh law ")
        content = content.replace("'law ", "'./run.sh law ")
        content = content.replace('"law ', '"./run.sh law ')
        content = content.replace("\nlaw ", "\n./run.sh law ")

        return content

    @property
    def slurm_job_file_factory_defaults(self):
        """Set job file factory defaults to use shared directory."""
        job_dir = os.path.expandvars("$HOME/tmp/law_jobs/ntupelize")
        os.makedirs(job_dir, exist_ok=True)
        return {
            "dir": job_dir,
            "mkdtemp": False,
            "render_variables": {
                "law_setup": "source ~/ml-tau/ml-tau-data/.law-venv/bin/activate"
            },
            "custom_content": "\n# Setup law environment\nsource ~/ml-tau/ml-tau-data/.law-venv/bin/activate\n\n",
        }

    def htcondor_log_directory(self):
        """Alias for slurm_output_directory (law bug workaround)."""
        return self.slurm_output_directory()


# =============================================================================
# All samples ntupelization task
# =============================================================================


class NtupelizeAllSamples(NtupelizeBaseTask, law.WrapperTask):
    """
    Ntupelize all configured samples.

    Parameters:
        config_type: Config type ('old' or 'new')
        job_workflow: Workflow for jobs ('slurm' or 'local')
    """

    def requires(self):
        """Require all samples to be processed."""
        return [
            NtupelizeSample(
                sample_name=sample.name,
                config_type=self.config_type,
                workflow=self.job_workflow,
                use_container=self.use_container,
            )
            for sample in self.workflow_config.get_samples_to_process()
        ]


# =============================================================================
# Validation tasks
# =============================================================================


class ValidateSample(NtupelizeBaseTask, SlurmWorkflow, law.LocalWorkflow):
    """
    Validate ntupelized output for a sample.

    This is a single-branch workflow that can run on SLURM or locally.

    Parameters:
        sample_name: Name of the sample to validate
        config_type: Config type ('old' or 'new')
        job_workflow: Workflow for jobs ('slurm' or 'local')
    """

    sample_name = luigi.Parameter(description="Name of the sample to validate")

    def create_branch_map(self):
        """Single branch for validation."""
        return {0: self.sample_name}

    def workflow_requires(self):
        """Require sample to be ntupelized first (all branches complete)."""
        # Request all branches of the NtupelizeSample workflow
        return {
            "ntupelize": NtupelizeSample(
                sample_name=self.sample_name,
                config_type=self.config_type,
                workflow=self.job_workflow,
                use_container=self.use_container,
            )
        }

    def output(self):
        """Define validation output directory."""
        validation_dir = self.workflow_config.validation_output_dir
        if validation_dir is None:
            validation_dir = os.path.expandvars("$HOME/ntuple_validation")

        output_dir = os.path.join(validation_dir, self.sample_name)
        # Use a marker file to indicate completion
        return law.LocalFileTarget(os.path.join(output_dir, "_validation_complete"))

    def run(self):
        """Run validation on the sample."""
        from ntupelizer.validation.validation import validate_ntupelizer_output

        sample = self.workflow_config.get_sample(self.sample_name)

        # Load all parquet files for this sample
        parquet_pattern = os.path.join(sample.output_dir, "*.parquet")
        parquet_files = sorted(glob.glob(parquet_pattern))

        if not parquet_files:
            raise RuntimeError(f"No parquet files found in {sample.output_dir}")

        print(f"Loading {len(parquet_files)} parquet files for validation")

        # Load and concatenate all data
        all_data = []
        for pq_file in parquet_files:
            data = ak.from_parquet(pq_file)
            all_data.append(data)

        data = ak.concatenate(all_data)
        print(f"Loaded {len(data)} jets for validation")

        # Create output directory
        output_dir = os.path.dirname(self.output().path)
        os.makedirs(output_dir, exist_ok=True)

        # Run validation
        figures = validate_ntupelizer_output(
            data=data,
            label=self.sample_name,
            output_dir=output_dir,
            show_plots=False,
        )

        print(f"Validation complete. Plots saved to {output_dir}")

        # Save summary statistics
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Sample: {self.sample_name}\n")
            f.write(f"Total jets: {len(data)}\n")
            f.write(f"Fields: {data.fields}\n")
            f.write(f"Plots generated: {list(figures.keys())}\n")

        # Touch the marker file
        self.output().touch()

    def slurm_resources(self, job_num, branches):
        """Define SLURM resources for validation job."""
        return {
            "partition": self.workflow_config.slurm.partition,
            "walltime": self.workflow_config.slurm.walltime,
            "memory": str(self.workflow_config.slurm.memory),
        }

    def slurm_output_directory(self):
        """Define directory for SLURM job logs."""
        validation_dir = self.workflow_config.validation_output_dir
        if validation_dir is None:
            validation_dir = os.path.expandvars("$HOME/ntuple_validation")
        log_dir = os.path.join(
            validation_dir, "slurm_logs", "validate", self.sample_name
        )
        return law.LocalDirectoryTarget(log_dir)

    def slurm_job_file_content(self):
        """Override job script to run law commands inside container."""
        import os

        content = super().slurm_job_file_content()

        # Replace law commands to run inside container where environment works
        # This avoids the venv symlink issue completely
        content = content.replace(" law ", " ./run.sh law ")
        content = content.replace("'law ", "'./run.sh law ")
        content = content.replace('"law ', '"./run.sh law ')
        content = content.replace("\nlaw ", "\n./run.sh law ")

        return content

    @property
    def slurm_job_file_factory_defaults(self):
        """Set job file factory defaults to use shared directory."""
        job_dir = os.path.expandvars("$HOME/tmp/law_jobs/validate")
        os.makedirs(job_dir, exist_ok=True)
        return {
            "dir": job_dir,
            "mkdtemp": False,
            "render_variables": {
                "law_setup": "source ~/ml-tau/ml-tau-data/.law-venv/bin/activate"
            },
            "custom_content": "\n# Setup law environment\nsource ~/ml-tau/ml-tau-data/.law-venv/bin/activate\n\n",
        }

    def htcondor_log_directory(self):
        """Alias for slurm_output_directory (law bug workaround)."""
        return self.slurm_output_directory()


class ValidateAllSamples(NtupelizeBaseTask, law.WrapperTask):
    """
    Validate all configured samples.

    Parameters:
        config_type: Config type ('old' or 'new')
        job_workflow: Workflow for jobs ('slurm' or 'local')
    """

    def requires(self):
        """Require all samples to be validated."""
        return [
            ValidateSample(
                sample_name=sample.name,
                config_type=self.config_type,
                workflow=self.job_workflow,
                use_container=self.use_container,
            )
            for sample in self.workflow_config.get_samples_to_process()
        ]


class CompareValidation(NtupelizeBaseTask, SlurmWorkflow, law.LocalWorkflow):
    """
    Create comparison validation plots across all samples.

    This is a single-branch workflow that can run on SLURM or locally.

    Parameters:
        config_type: Config type ('old' or 'new')
        job_workflow: Workflow for jobs ('slurm' or 'local')
    """

    def create_branch_map(self):
        """Single branch for comparison."""
        return {0: "comparison"}

    def workflow_requires(self):
        """Require all samples to be validated first."""
        return {
            "validate": ValidateAllSamples(
                config_type=self.config_type,
                job_workflow=self.job_workflow,
                use_container=self.use_container,
            )
        }

    def output(self):
        """Define comparison output directory."""
        validation_dir = self.workflow_config.validation_output_dir
        if validation_dir is None:
            validation_dir = os.path.expandvars("$HOME/ntuple_validation")

        output_dir = os.path.join(validation_dir, "comparison")
        return law.LocalFileTarget(os.path.join(output_dir, "_comparison_complete"))

    def run(self):
        """Create comparison plots."""
        from ntupelizer.validation.validation import validate_multiple_datasets

        datasets = {}
        for sample in self.workflow_config.get_samples_to_process():
            parquet_pattern = os.path.join(sample.output_dir, "*.parquet")
            parquet_files = sorted(glob.glob(parquet_pattern))

            if parquet_files:
                all_data = [ak.from_parquet(f) for f in parquet_files]
                datasets[sample.name] = ak.concatenate(all_data)

        if len(datasets) < 2:
            print("Need at least 2 samples for comparison")
            self.output().touch()
            return

        output_dir = os.path.dirname(self.output().path)
        os.makedirs(output_dir, exist_ok=True)

        validate_multiple_datasets(
            datasets=datasets,
            output_dir=output_dir,
            show_plots=False,
        )

        print(f"Comparison plots saved to {output_dir}")
        self.output().touch()

    def slurm_resources(self, job_num, branches):
        """Define SLURM resources for comparison job."""
        return {
            "partition": self.workflow_config.slurm.partition,
            "walltime": self.workflow_config.slurm.walltime,
            "memory": str(self.workflow_config.slurm.memory),
        }

    def slurm_output_directory(self):
        """Define directory for SLURM job logs."""
        validation_dir = self.workflow_config.validation_output_dir
        if validation_dir is None:
            validation_dir = os.path.expandvars("$HOME/ntuple_validation")
        log_dir = os.path.join(validation_dir, "slurm_logs", "comparison")
        return law.LocalDirectoryTarget(log_dir)

    def slurm_job_file_content(self):
        """Override job script to run law commands inside container."""
        import os

        content = super().slurm_job_file_content()

        # Replace law commands to run inside container where environment works
        # This avoids the venv symlink issue completely
        content = content.replace(" law ", " ./run.sh law ")
        content = content.replace("'law ", "'./run.sh law ")
        content = content.replace('"law ', '"./run.sh law ')
        content = content.replace("\nlaw ", "\n./run.sh law ")

        return content

    @property
    def slurm_job_file_factory_defaults(self):
        """Set job file factory defaults to use shared directory."""
        job_dir = os.path.expandvars("$HOME/tmp/law_jobs/comparison")
        os.makedirs(job_dir, exist_ok=True)
        return {
            "dir": job_dir,
            "mkdtemp": False,
            "render_variables": {
                "law_setup": "source ~/ml-tau/ml-tau-data/.law-venv/bin/activate"
            },
            "custom_content": "\n# Setup law environment\nsource ~/ml-tau/ml-tau-data/.law-venv/bin/activate\n\n",
        }

    def htcondor_log_directory(self):
        """Alias for slurm_output_directory (law bug workaround)."""
        return self.slurm_output_directory()


# =============================================================================
# Full pipeline task
# =============================================================================


class FullPipeline(NtupelizeBaseTask, law.WrapperTask):
    """
    Run the full ntupelization and validation pipeline.

    This task:
    1. Ntupelizes all configured samples (on SLURM by default)
    2. Validates each sample
    3. Creates comparison plots

    Parameters:
        config_type: Config type ('old' or 'new')
        job_workflow: Workflow for jobs ('slurm' or 'local')
        skip_comparison: Skip comparison plots (default: False)
    """

    skip_comparison = luigi.BoolParameter(
        default=False,
        description="Skip generating comparison plots",
    )

    def requires(self):
        """Require validation and comparison (ntupelization is implicit via ValidateSample)."""
        # ValidateAllSamples -> ValidateSample -> NtupelizeSample (chain)
        # So ntupelization happens first automatically
        if self.skip_comparison:
            return ValidateAllSamples(
                config_type=self.config_type,
                job_workflow=self.job_workflow,
                use_container=self.use_container,
            )
        else:
            # CompareValidation -> ValidateAllSamples -> ValidateSample -> NtupelizeSample
            return CompareValidation(
                config_type=self.config_type,
                workflow=self.job_workflow,
                use_container=self.use_container,
            )
