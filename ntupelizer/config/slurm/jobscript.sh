#!/bin/bash
# Snakemake cluster jobscript.
#
# Explicitly activates the py39 venv (which has snakemake installed) before
# running the rule. The rule's shell block calls "apptainer exec ..." for the
# actual data processing inside the container.

# properties = {properties}

{exec_job}
