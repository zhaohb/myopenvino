#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for working with processes.
"""

import logging
import subprocess


def cmd_exec(args, env=None, log=None, verbose=True):
    """ Run cmd using subprocess with logging and other improvements
    """
    if log is None:
        log = logging.getLogger()
    log_out = log.info if verbose else log.debug

    log_out(  # pylint: disable=logging-fstring-interpolation
        f'========== cmd: {" ".join(args)}'
    )

    proc = subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        universal_newlines=True,
    )
    output = []
    for line in iter(proc.stdout.readline, ""):
        log_out(line.strip("\n"))
        output.append(line)
        if line or proc.poll() is None:
            continue
        break
    outs = proc.communicate()[0]

    if outs:
        log_out(outs.strip("\n"))
        output.append(outs)
    log_out("========== Completed. Exit code: %d", proc.returncode)
    return proc.returncode, "".join(output)
