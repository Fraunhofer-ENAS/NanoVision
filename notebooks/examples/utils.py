import os
from pathlib import Path
import subprocess
import sys


def run_module(module: str, *args: str) -> None:
    """
    Run a NanoVision module and stream its output into the notebook.

    The current notebook kernel's Python interpreter is used so that
    the runner executes in the same environment as the notebook.
    """
    command = [
        sys.executable,
        "-u",
        "-m",
        module,
        *map(str, args),
    ]

    print("Running:")
    print(" ".join(command))
    print("-" * 80)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )

    print("-" * 80)
    print("Runner completed successfully.")