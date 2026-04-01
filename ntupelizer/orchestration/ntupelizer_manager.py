import os
import glob
import hydra
from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig
from pathlib import Path
from typing import List, Dict, Any
import subprocess, time
import tqdm
import awkward as ak
import numpy as np


def job_finished(job_id):
    r = subprocess.run(
        ["squeue", "-j", job_id, "-h"], stdout=subprocess.PIPE, text=True, check=True
    )
    return r.stdout.strip() == ""


def monitor_jobs(job_ids: List[int]):
    jobs_finished = 0
    while jobs_finished < len(job_ids):
        jobs_finished = sum([job_finished(job_id) for job_id in job_ids])
        print(f"[{jobs_finished}/{len(job_ids)}] jobs finished")
        time.sleep(10)


def split_train_test(data, split=0.8):
    ndata = len(data)
    ntrain = int(ndata * split)
    data_train = data[:ntrain]
    data_test = data[ntrain:]
    print(f"N={ndata}, Ntrain={len(data_train)} Ntest={len(data_test)}")
    return data_train, data_test


def merge_files(output_dir: str, sample_name: str, columns=None):
    ntupelized_paths = glob.glob(os.path.join(output_dir, "*.parquet"))
    print(f"Merging ntupelized files for =={sample_name}== sample")
    data = []
    for fi in tqdm.tqdm(list(ntupelized_paths)):
        ret = ak.from_parquet(fi, columns=columns)
        ret = ak.Array({k: ret[k] for k in ret.fields})
        data.append(ret)
    data = ak.concatenate(data)

    # shuffle data
    perm = np.random.permutation(len(data))
    data = data[perm]
    data_train, data_test = split_train_test(data)
    test_path = os.path.join(output_dir, f"{sample_name}_test.parquet")
    train_path = os.path.join(output_dir, f"{sample_name}_train.parquet")
    ak.to_parquet(data_test, test_path, row_group_size=1024)
    ak.to_parquet(data_train, train_path, row_group_size=1024)


class NtupelizerOrchestrator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.orchestration_dir = Path(__file__).parent
        self.package_dir = self.orchestration_dir.parent.parent
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.orchestration_dir / "templates")
        )

    def process_sample(self, sample_name: str):
        input_dir = self.cfg.samples[sample_name].input_dir
        output_dir = self.cfg.samples[sample_name].output_dir
        is_signal = self.cfg.samples[sample_name].is_signal

        input_file_list = list(glob.glob(os.path.join(input_dir, "*.root")))
        job_ids = []
        for input_path in input_file_list:
            job_id = self.submit_slurm_job(
                input_path=input_path, output_dir=output_dir, is_signal=is_signal
            )
            job_ids.append(job_id)
        monitor_jobs(job_ids=job_ids)
        merge_files(output_dir=output_dir, sample_name=sample_name)

    def run(self):
        datasets = self.cfg.samples.keys()
        for dataset in datasets:
            self.process_sample(sample_name=dataset)

    def submit_slurm_job(
        self,
        input_path: str,
        output_dir: str,
        is_signal: bool,
    ) -> str:
        """Submit a SLURM job for processing a chunk of files."""
        basename = os.path.basename(input_path).split(".")[0]

        output_path = os.path.join(output_dir, f"{basename}.parquet")
        job_dir = os.path.join(output_dir, "submission_scripts")
        err_dir = os.path.join(output_dir, "error_files")
        out_dir = os.path.join(output_dir, "out_files")
        os.makedirs(job_dir, exist_ok=True)
        # Create job script
        job_script_path = os.path.join(job_dir, f"{basename}.sh")

        slurm_cfg = self.cfg.get("slurm", {})
        partition = slurm_cfg.get("partition", "short")
        walltime = slurm_cfg.get("walltime", "00:15:00")
        memory = slurm_cfg.get("memory", 8000)
        cpus = slurm_cfg.get("cpus_per_task", 1)
        ntasks = slurm_cfg.get("ntasks", 1)
        job_name = slurm_cfg.get("job_name", "tau_ntupelizer")

        template = self.jinja_env.get_template("slurm_job.sh.j2")
        job_script_content = template.render(
            job_name=job_name,
            job_dir=job_dir,
            err_dir=err_dir,
            out_dir=out_dir,
            partition=partition,
            walltime=walltime,
            memory=memory,
            cpus=cpus,
            ntasks=ntasks,
            working_dir=self.orchestration_dir.parent.parent,
            environment_script="/home/laurits/ml-tau/ml-tau-data/run.sh",
            processing_script="/home/laurits/ml-tau/ml-tau-data/ntupelizer/scripts/ntupelize.py",
            input_path=input_path,
            output_path=output_path,
            is_signal=is_signal,
        )

        with open(job_script_path, "wt") as f:
            f.write(job_script_content)

        # Submit job
        cmd = ["sbatch", str(job_script_path)]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True,
            )
            job_id = result.stdout.strip().split()[-1]
            return job_id

        except subprocess.CalledProcessError as e:
            print("sbatch failed!")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)


@hydra.main(
    version_base=None, config_path="../config", config_name="podio_root_ntupelizer"
)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    orchestrator = NtupelizerOrchestrator(cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
