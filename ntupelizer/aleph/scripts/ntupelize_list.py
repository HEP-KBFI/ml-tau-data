import os
import time
import hydra
from omegaconf import DictConfig
from ntupelizer.aleph.tools import ntupelize_aleph as na

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    input_files = cfg.input_paths
    output_files = cfg.output_paths
    start_time = time.time()
    for input_path, output_path in zip(input_files, output_files):
        na.ntupelize_file(input_path=input_path, output_path=output_path)
    end_time = time.time()
    print(f"Finished processing {len(input_files)} files in {end_time-start_time} s.")


if __name__ == "__main__":
    main()
