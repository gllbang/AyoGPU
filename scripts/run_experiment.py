"""Persist a long experiment's log and exit status across terminal interruptions."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = (
        args.command[1:] if args.command and args.command[0] == "--" else args.command
    )
    if not command:
        parser.error("A command is required after --")
    directory = Path(args.job_dir).resolve()
    if args.supervise:
        start = time.time()
        try:
            process = subprocess.Popen(command)
            (directory / "process.json").write_text(
                json.dumps(
                    {"pid": process.pid, "command": command, "started_at": start},
                    indent=2,
                )
                + "\n"
            )
            code = process.wait()
        except Exception as exc:
            print(f"Experiment launch failed: {exc}", flush=True)
            code = 1
        (directory / "exit.json").write_text(
            json.dumps({"exit_code": code, "elapsed_s": time.time() - start}, indent=2)
            + "\n"
        )
        return code
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--supervise",
                "--job-dir",
                str(directory),
                "--",
                *command,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(
        json.dumps({"supervisor_pid": process.pid, "job_dir": str(directory)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
