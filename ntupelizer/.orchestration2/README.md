# Ntupelization Orchestration

This directory contains a complete orchestration system for ntupelizing HEP datasets with SLURM support. The system supports both legacy and modern (EDM4HEP/PODIO) data formats.

## Basic Usage

**Run complete pipeline with legacy format:**
```bash
python run_pipeline.py --config-path=../config --config-name=ntupelizer \
  config_choice=old \
  input_dir=/path/to/your/input/files \
  output_dir=/path/to/your/output
```

**Run complete pipeline with EDM4HEP/PODIO format:**
```bash
python run_pipeline.py --config-path=../config --config-name=ntupelizer \
  config_choice=new \
  input_dir=/path/to/your/input/files \
  output_dir=/path/to/your/output
```

## Ntupelization Only

If you only want to run the ntupelization step (skip merging, weights, validation):

```bash
python ntupelize_files.py --config-path=../config --config-name=ntupelizer \
  config_choice=old \
  input_dir=/path/to/input \
  output_dir=/path/to/output
```

## Pipeline Overview

The complete pipeline consists of four steps:

1. **Ntupelization**: Process ROOT files and convert to parquet format
2. **Merging**: Combine all parquet files for each sample  
3. **Weight assignment**: Add kinematic and class reweighting
4. **Validation**: Create validation plots and reports

## Configuration

### Master Configuration

The system uses a master config that allows switching between formats:
- `../config/ntupelizer.yaml` - Main selector config
- `../config/ntupelizer_base/old.yaml` - Legacy ntupelizer settings
- `../config/ntupelizer_base/new.yaml` - EDM4HEP/PODIO settings

### Advanced Configuration

```bash
# Limit files per dataset and adjust batch size
python run_pipeline.py --config-path=../config --config-name=ntupelizer \
  config_choice=old \
  input_dir=/data/samples \
  output_dir=/output/ntuples \
  max_files_per_dataset=50 \
  batch_size=50000

# Customize SLURM job parameters
python run_pipeline.py --config-path=../config --config-name=ntupelizer \
  config_choice=new \
  input_dir=/data \
  output_dir=/output \
  slurm_config.memory=8G \
  slurm_config.time=04:00:00 \
  slurm_config.partition=long

# Enable validation plots
python run_pipeline.py --config-path=../config --config-name=ntupelizer \
  config_choice=old \
  input_dir=/data \
  output_dir=/output \
  enable_validation=true
```

## Individual Pipeline Steps

Run each step separately:

```bash
# 1. Only ntupelize files
python ntupelize_files.py --config-path=../config --config-name=ntupelizer \
  config_choice=old \
  input_dir=/data \
  output_dir=/output

# 2. Only merge existing parquet files  
python merge_files.py --config-path=../config --config-name=ntupelizer \
  output_dir=/output

# 3. Only add weights to merged files
python add_weights.py --config-path=../config --config-name=ntupelizer \
  output_dir=/output

# 4. Only create validation plots
python validate.py --config-path=../config --config-name=ntupelizer \
  output_dir=/output
```

## SLURM Support

SLURM is automatically used when the system detects a SLURM environment. Job parameters can be customized:

- Job scripts are generated in `/tmp/slurm_jobs/` using Jinja2 templates
- Supports parallel processing with configurable resources
- Automatic container detection and usage via `./run.sh` wrapper

## Output Structure

- Individual parquet files: `{output_dir}/{basename}.parquet`
- Merged files: `{output_dir}/{sample_name}_merged.parquet`
- Weighted files: `{output_dir}/{sample_name}_weighted.parquet`
- Validation plots and reports: `$HOME/ntuple_validation/`

## Scripts

- `run_pipeline.py`: Main orchestrator that runs all steps
- `ntupelize_files.py`: Manages ntupelization jobs (local or SLURM)
- `ntupelize_batch.sh`: Bash script that processes individual file batches
- `merge_files.py`: Merges parquet files for each sample
- `add_weights.py`: Adds sample and event weights
- `validate.py`: Creates validation plots and summary report