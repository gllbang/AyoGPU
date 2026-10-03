"""Download the public Meta-Llama-3-8B snapshot with per-file SHA-256 checks."""

import argparse
import hashlib
import json
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def checksum(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/disk_n/lsz/model/Meta-Llama-3-8B")
    args = parser.parse_args()
    model = "LLM-Research/Meta-Llama-3-8B"
    api = f"https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true"
    response = subprocess.check_output(
        ["curl", "-fsSL", "--max-time", "30", api], text=True
    )
    metadata = json.loads(response)
    if metadata.get("Code") != 200:
        raise RuntimeError(f"Model metadata fetch failed: {metadata.get('Message')}")
    entries = [
        entry
        for entry in metadata["Data"]["Files"]
        if entry["Type"] == "blob"
        and "/" not in entry["Path"]
        and entry["Name"] != ".gitattributes"
    ]
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {len(entries)} files, {sum(e['Size'] for e in entries) / 1e9:.2f} GB to {directory}",
        flush=True,
    )

    def download(entry):
        path = directory / entry["Name"]
        if path.exists():
            if (
                path.stat().st_size == entry["Size"]
                and checksum(path) == entry["Sha256"]
            ):
                print(f"Verified existing {path.name}", flush=True)
                return
            raise FileExistsError(f"Existing file differs from public snapshot: {path}")
        url = f"https://modelscope.cn/models/{model}/resolve/{entry['Revision']}/{urllib.parse.quote(entry['Path'])}"
        partial = path.with_name(path.name + ".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--connect-timeout",
                "15",
                "--max-time",
                "1800",
                "--continue-at",
                "-",
                url,
                "--output",
                str(partial),
            ],
            check=True,
        )
        if (
            partial.stat().st_size != entry["Size"]
            or checksum(partial) != entry["Sha256"]
        ):
            raise ValueError(
                f"Downloaded file failed size/SHA-256 validation: {partial}"
            )
        partial.rename(path)
        print(f"Verified {path.name} ({entry['Size']} bytes)", flush=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, entries))
    (directory / "herasys_download_manifest.json").write_text(
        json.dumps({"model": model, "metadata_url": api, "files": entries}, indent=2)
        + "\n"
    )
    print("HERASYS_PAPER_MODEL_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
