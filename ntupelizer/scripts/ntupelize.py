import os
import time

import hydra
from omegaconf import DictConfig

from ntupelizer.tools import ntupelizing as nt

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


@hydra.main(
    version_base=None, config_path="../config", config_name="podio_root_ntupelizer"
)
def main(cfg: DictConfig) -> None:
    ntupelizer_cls = getattr(nt, cfg.ntupelizer_class)
    processor = ntupelizer_cls(cfg)
    if not os.path.exists(cfg.output_path):
        start_time = time.time()
        processor.ntupelize(
            input_path=cfg.input_path,
            output_path=cfg.output_path,
            signal_sample=cfg.is_signal,
        )
        end_time = time.time()
        print(f"Finished processing in {end_time - start_time} s.")
    else:
        print("File already processed, skipping.")


if __name__ == "__main__":
    main()
