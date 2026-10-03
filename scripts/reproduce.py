"""Reproduce the measured HeraSys/Ayo configuration without changing old results."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(config, root=ROOT):
    if sha256(root / config["dataset"]) != config["dataset_sha256"]:
        raise ValueError("Dataset differs from the measured configuration")
    for name in ("herasys", "ayo"):
        system = config[name]
        if sha256(root / system["snapshot"]) != system["snapshot_sha256"]:
            raise ValueError(f"{system['system']} source archive changed")
        env = json.loads(
            (root / system["recorded_results"] / "environment.json").read_text()
        )
        source_id = hashlib.sha256(
            json.dumps(env["source_sha256"], sort_keys=True).encode()
        ).hexdigest()
        if source_id != system["source_id"]:
            raise ValueError(f"{system['system']} source manifest changed")
        with tarfile.open(root / system["snapshot"]) as archive:
            for path, expected in env["source_sha256"].items():
                stream = archive.extractfile(path)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                    raise ValueError(f"Recorded source differs: {path}")
        if name == "ayo":
            checkout = root / system["checkout"]
            for path, expected in env["source_sha256"].items():
                if sha256(checkout / path) != expected:
                    raise ValueError(f"Ayo baseline source differs: {path}")
            if sha256(root / system["runner"]) != system["runner_sha256"]:
                raise ValueError("Recorded Ayo runner changed")
    system = config["herasys"]
    if sha256(root / system["profiles"]) != system["profiles_sha256"]:
        raise ValueError("Recorded latency profiles changed")


def cli_arguments(arguments):
    result = []
    for name, value in arguments.items():
        result.append("--" + name.replace("_", "-"))
        result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def build_commands(config, output, source_root, source="measured", root=ROOT):
    hers, ayo = config["herasys"], config["ayo"]
    variant = hers["recorded_variant"] if source == "measured" else "HeraSys"
    baseline = ayo["recorded_variant"] if source == "measured" else "Ayo"
    commands = [
        [sys.executable, "-m", "benchmarks.herasys"]
        + cli_arguments(hers["arguments"])
        + [
            "--variants", variant,
            "--dataset", str(root / config["dataset"]),
            "--profiles", str(root / hers["profiles"]),
            "--output", str(output / "HeraSys"),
        ],
        [sys.executable, str(source_root / "scripts/reproduce_ayo_full.py")]
        + cli_arguments(ayo["arguments"])
        + [
            "--upstream", str(root / ayo["checkout"]),
            "--dataset", str(root / config["dataset"]),
            "--output", str(output / "Ayo"),
        ],
    ]
    for name in ("HeraSys", "Ayo"):
        commands.append([
            sys.executable, str(root / "scripts/verify_herasys_results.py"),
            str(output / name),
        ])
    commands.append([
        sys.executable, str(root / "scripts/report_herasys_suite.py"),
        str(output / "HeraSys"), str(output / "Ayo"),
        "--variant", variant, "--baselines", baseline,
        "--variant-label", "HeraSys", "--baseline-label", "Ayo",
        "--output", str(output / "comparison"), "--plot", "--table-only",
    ])
    return commands


def extract_snapshot(snapshot, destination):
    with tarfile.open(snapshot) as archive:
        members = archive.getmembers()
        for member in members:
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination.resolve()) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination, members=members)


def save_source_snapshot(config, output, source_root, source, root=ROOT):
    target = output / "HeraSys/source.tar.gz"
    if source == "measured":
        shutil.copy2(root / config["herasys"]["snapshot"], target)
        return
    with tarfile.open(target, "w:gz") as archive:
        for name in ("HeraSys", "benchmarks", "scripts", "tests", "patches", "setup.py", "pyproject.toml"):
            archive.add(
                source_root / name, arcname=name,
                filter=lambda member: None if "__pycache__" in Path(member.name).parts else member,
            )


def require_idle_gpus():
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index", "--format=csv,noheader",
    ], text=True).splitlines()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    if len(devices) != 4 or visibility not in ("all", ",".join(devices)):
        raise RuntimeError("The measured configuration requires all four GPUs")
    running = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
    ], text=True).strip()
    if running:
        raise RuntimeError("GPUs are occupied; existing processes will not be stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/rag_rerank.json")
    parser.add_argument("--source", choices=["measured", "current"], default="measured")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify_inputs(config)
    if args.verify_only:
        print("HeraSys/Ayo source archives, baseline checkout, profiles and dataset verified.")
        return
    if args.output is None:
        parser.error("A new output directory is required")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output directory already exists; previous results are preserved")
    if args.dry_run:
        source_root = ROOT if args.source == "current" else ROOT / ".runtime/herasys-measured"
        for command in build_commands(config, output, source_root, args.source):
            print(shlex.join(command))
        return
    if not os.environ.get("HERASYS_PG_DSN"):
        parser.error("Set HERASYS_PG_DSN after starting scripts/start_postgres.sh")
    require_idle_gpus()
    for key in ("llm_model", "embedding_model", "reranker_model"):
        if not Path(config["herasys"]["arguments"][key]).is_dir():
            parser.error(f"Missing model directory: {key}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.json").write_text(
        json.dumps({"source": args.source, "configuration": config}, indent=2) + "\n"
    )
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="herasys-reproduce-", dir=runtime) as directory:
        source_root = ROOT
        if args.source == "measured":
            source_root = Path(directory)
            extract_snapshot(ROOT / config["herasys"]["snapshot"], source_root)
            shutil.copy2(
                ROOT / config["ayo"]["runner"], source_root / "scripts/reproduce_ayo_full.py"
            )
        env = {**os.environ, "PYTHONPATH": str(source_root)}
        for index, command in enumerate(build_commands(config, output, source_root, args.source)):
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=source_root, env=env, check=True)
            if index == 0:
                save_source_snapshot(config, output, source_root, args.source)


if __name__ == "__main__":
    main()
