"""Merge per-file parquets for one sample, shuffle, and split into train/test.

Usage:
    merge_files.py -i <input_dir> -o <output_dir> -s <sample_shortname> -f <train_frac>

Options:
    -i <input_dir>         Directory containing the per-file .parquet inputs.
    -o <output_dir>        Directory where <sample_shortname>_train.parquet and
                           <sample_shortname>_test.parquet will be written.
    -s <sample_shortname>  Short identifier for the sample (e.g. "qq", "zh", "z").
    -f <train_frac>        Fraction of events to assign to the train split (e.g. 0.7).
"""

import glob
import os
import tqdm
import numpy as np
import awkward as ak
from docopt import docopt


def load_sample(path):
    columns = [
        # basic reco inputs
        "reco_jet_p4",
        "reco_cand_p4s",
        "reco_cand_charges",
        "reco_cand_pdgs",
        # advanced reco inputs: track impact parameters
        "reco_cand_dz",
        "reco_cand_dz_error",
        "reco_cand_dxy",
        "reco_cand_dxy_error",
        # targets
        "gen_jet_p4",  # generated jet p4
        "gen_jet_tau_p4",  # tau visible momentum, excluding neutrino
        "gen_jet_tau_decaymode",  # tau decay mode
        "gen_jet_tau_charge",
    ]
    p4_fields = ["rho", "phi", "eta", "t"]
    data = []
    for fi in tqdm.tqdm(list(glob.glob(path + "/*.parquet"))):
        ret = ak.from_parquet(fi, columns=columns)
        # Normalize gen_jet_tau_p4 to the common 4-field schema {rho, phi, eta, t}.
        # Some files have extra fields (x, y, z, tau), which causes ak.concatenate
        # to produce a dense_union type that pyarrow cannot write to parquet.
        ret = ak.Array(
            {
                **{k: ret[k] for k in columns if k != "gen_jet_tau_p4"},
                "gen_jet_tau_p4": ak.zip(
                    {f: ret["gen_jet_tau_p4"][f] for f in p4_fields}
                ),
            }
        )
        data.append(ret)
    data = ak.concatenate(data)
    print("Fields before merge: ", data.fields)

    # shuffle data
    perm = np.random.permutation(len(data))
    data = data[perm]

    return data


def split_train_test(data, split=0.8):
    ndata = len(data)
    ntrain = int(ndata * split)
    data_train = data[:ntrain]
    data_test = data[ntrain:]
    print(f"N={ndata}, Ntrain={len(data_train)} Ntest={len(data_test)}")
    return data_train, data_test


if __name__ == "__main__":
    args = docopt(__doc__)

    sample_shortname = args["-s"]
    input_dir = args["-i"]
    train_frac = float(args["-f"])
    output_dir = args["-o"]

    os.makedirs(output_dir, exist_ok=True)

    data = load_sample(input_dir)
    data_train, data_test = split_train_test(data, split=train_frac)

    ak.to_parquet(
        data_train,
        os.path.join(output_dir, f"{sample_shortname}_train.parquet"),
        row_group_size=1024,
    )
    ak.to_parquet(
        data_test,
        os.path.join(output_dir, f"{sample_shortname}_test.parquet"),
        row_group_size=1024,
    )
