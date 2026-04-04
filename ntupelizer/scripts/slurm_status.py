#!/usr/bin/env python3
"""Check status of a SLURM job for Snakemake's --cluster-status hook."""

import subprocess
import sys


def main():
    job_id = sys.argv[1]

    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
        capture_output=True, text=True, timeout=30,
    )

    state = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""

    if state in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}:
        print("running")
    elif state == "COMPLETED":
        print("success")
    else:
        # FAILED, CANCELLED, TIMEOUT, NODE_FAIL, PREEMPTED, etc.
        print("failed")


if __name__ == "__main__":
    main()
