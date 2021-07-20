#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test conditional compilation statistics collection.
"""

import glob
import os
import sys

from proc_utils import cmd_exec  # pylint: disable=import-error


def test_cc_collect(test_id, model, sea_runtool, benchmark_app, collector_dir, artifacts):
    """ Test conditional compilation statistics collection
    """
    out = artifacts / test_id
    # cleanup old data if any
    prev_results = glob.glob(f"{out}.pid*.csv")
    for path in prev_results:
        os.remove(path)
    # run use case
    returncode, output = cmd_exec(
        [
            sys.executable,
            str(sea_runtool),
            f"--output={out}",
            f"--bindir={collector_dir}",
            "--app_status",
            "!",
            str(benchmark_app),
            "-d=CPU",
            f"-m={model}",
            "-niter=1",
            "-nireq=1",
        ]
    )
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
    assert (
        len(glob.glob(f"{out}.pid*.csv")) == 1
    ), f'Multiple or none "{out}.pid*.csv" files'
