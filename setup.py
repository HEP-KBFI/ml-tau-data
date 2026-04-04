"""Setup script for ml-tau-data package."""

from setuptools import setup, find_packages

# Core dependencies for running the workflow orchestration
# (minimal set needed for law/luigi scheduling)
CORE_DEPS = [
    "law",
    "luigi",
    "jinja2",
    "omegaconf",
    "snakemake>=7,<8",
    "pulp>=2.7,<3",
]

# Full dependencies for running the actual ntupelization
FULL_DEPS = CORE_DEPS + [
    "vector",
    "awkward",
    "numpy",
    "particle",
    "fastjet",
    "uproot",
    "hydra-core",
    "numba",
    "pyarrow",
    "matplotlib",
]

setup(
    name="ntupelizer",
    version="0.1.0",
    description="Tau jet ntupelization tools for ML-tau project",
    author="HEP-KBFI",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ntupelizer.orchestration": ["templates/*.j2"],
        "ntupelizer.config": ["*.yaml"],
    },
    python_requires=">=3.8",
    install_requires=CORE_DEPS,
    extras_require={
        "full": FULL_DEPS,
        "core": CORE_DEPS,
    },
    entry_points={
        "console_scripts": [
            "ntupelizer-workflow=ntupelizer.orchestration.run_workflow:main",
        ],
    },
)
