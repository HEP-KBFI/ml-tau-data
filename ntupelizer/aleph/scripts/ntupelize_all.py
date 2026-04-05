import os
import glob
import hydra
import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig
from ntupelizer.aleph.tools import ntupelize_aleph as na

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

INPUT_DIR = "/local/laurits/ALEPH/1994_old/LAST"
OUTPUT_DIR = "/local/laurits/ALEPH/1994_old_ntupelized"

orchestration_dir = Path(__file__).parent.parent
jinja_env = Environment(loader=FileSystemLoader(orchestration_dir / "templates"))


def submit_slurm_job(input_paths: str, output_dir: str, idx: int) -> str:
    """Submit a SLURM job for processing a chunk of files."""
    output_paths = []
    for input_path in input_paths:
        basename = os.path.basename(input_path).split(".")[0]
        output_path = os.path.join(output_dir, f"{basename}.parquet")
        output_paths.append(output_path)
    job_dir = os.path.join(output_dir, "submission_scripts")
    err_dir = os.path.join(output_dir, "error_files")
    out_dir = os.path.join(output_dir, "out_files")
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    # Create job script
    job_script_path = os.path.join(job_dir, f"chunk_{idx}.sh")

    input_paths_str = ",".join(input_paths)
    output_paths_str = ",".join(output_paths)

    template = jinja_env.get_template("ntupelize_files.sh.j2")
    job_script_content = template.render(
        job_name="aleph_ntupelizer",
        job_dir=job_dir,
        err_dir=err_dir,
        out_dir=out_dir,
        partition="main",
        walltime="01:00:00",
        memory=8000,
        cpus=1,
        ntasks=1,
        working_dir=orchestration_dir.parent.parent,
        environment_script="/home/laurits/ml-tau/ml-tau-data/run.sh",
        processing_script="/home/laurits/ml-tau/ml-tau-data/ntupelizer/aleph/scripts/ntupelize_list.py",
        input_paths=input_paths_str,
        output_paths=output_paths_str,
    )

    with open(job_script_path, "wt") as f:
        f.write(job_script_content)

    # # Submit job
    # cmd = ["sbatch", str(job_script_path)]
    # try:
    #     result = subprocess.run(
    #         cmd,
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,
    #         universal_newlines=True,
    #         check=True,
    #     )
    #     job_id = result.stdout.strip().split()[-1]
    #     return job_id

    # except subprocess.CalledProcessError as e:
    #     print("sbatch failed!")
    #     print("stdout:", e.stdout)
    #     print("stderr:", e.stderr)


def split_list_into_chunks(lst, num_chunks=20):
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        # Add 1 extra element to first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    input_wcp = os.path.join(INPUT_DIR, "*", "data_*.root")
    input_paths = list(glob.glob(input_wcp))
    job_chunks = split_list_into_chunks(input_paths, num_chunks=30)
    for i, input_chunk in enumerate(job_chunks):
        submit_slurm_job(input_paths=input_chunk, output_dir=OUTPUT_DIR, idx=i)


if __name__ == "__main__":
    main()
