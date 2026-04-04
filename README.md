# Data
In this repository you will find the data processing tools used in the machine learned hadronically decaying tau lepton reconstruction and identification project.
The dataset simulated within this project is further described in the 



## Future Dataset [![DOI](https://zenodo.org/badge/DOI-10.5281/zenodo.13881061.svg)](https://doi.org/10.5281/zenodo.13881061)

The dataset contains 2 signal samples (ZH->Ztautau and Z->tautau) and one background sample (Z->qq).
While the validation plots can be reproduced with [this script](notebooks/data_intro.ipynb), here is a selection of these:

The generator-level hadronically decaying tau visible transverse momentum:

<img src="images/gen_tau_visible_pt.png" width="50%"/>

The jet substructure of two neutral-hadronless decay modes:

<img src="images/jet_2D_shapes_ZH_DM0_DM3.png" width="100%"/>



### Setup the environment for law

```bash
scl enable rh-python38 bash
python3 -m venv .law-env
source .law-env/bin/activate
pip install --upgrade pip
pip install -e ".[full]"
```

If an existing Snakemake install fails with an error like
`AttributeError: module 'pulp' has no attribute 'list_solvers'`, the
environment has an incompatible `pulp` version. This repo expects:

```bash
pip install "snakemake>=7,<8" "pulp>=2.7,<3"
```

When running from the repo root, `sitecustomize.py` also adds a small
compatibility shim for older PuLP releases that expose `listSolvers()`
instead of `list_solvers()`.
