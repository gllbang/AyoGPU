import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest
from setuptools import find_packages

from HeraSys import HeraConfig, HeraSys
from HeraSys.backends import reference_pools
from benchmarks.herasys import parser
from scripts.reproduce import (
    build_commands,
    extract_snapshot,
    require_idle_gpus,
    save_source_snapshot,
    verify_inputs,
)

ROOT = Path(__file__).resolve().parents[2]


def test_primary_herasys_defaults_match_the_measured_scheduler():
    recorded = ROOT / "results/HeraSys/gpu_rag_rerank_HeraSys_c100_r0/summary.json"
    if recorded.exists():
        assert asdict(HeraConfig()) == json.loads(recorded.read_text())["config"]
    assert HeraConfig().contribution_scope == "direct"
    assert HeraConfig().max_query_wait_s == 2.0
    for variant in ("HeraSys", "herasys", "herasys_direct_aged"):
        assert HeraConfig.variant(variant) == HeraConfig()
    assert HeraSys(reference_pools()).config == HeraConfig()
    args = parser().parse_args(["--output", "unused"])
    assert args.variants == ["HeraSys"]
    assert (args.cache_entries, args.cache_ttl_s) == (4096, 300)


@pytest.mark.parametrize("name", ["herasys_unaged", "ayo_static", "fcfs", "srtf"])
def test_unaged_controls_and_calibration_do_not_inherit_query_aging(name):
    config = HeraConfig.variant(name)
    assert config.max_query_wait_s is None
    assert config.max_ready_wait_s is None


def test_recorded_configuration_and_current_commands_keep_systems_separate(tmp_path):
    config = json.loads((ROOT / "experiments/rag_rerank.json").read_text())
    if (ROOT / config["herasys"]["snapshot"]).exists() and (ROOT / "baselines/Ayo").exists():
        verify_inputs(config)
    for source in ("measured", "current"):
        commands = build_commands(config, tmp_path, ROOT, source)
        hers, ayo = commands[:2]
        expected = "herasys_direct_aged" if source == "measured" else "HeraSys"
        assert hers[hers.index("--variants") + 1] == expected
        assert "--batch-wait-s" in ayo
        assert ayo[ayo.index("--upstream") + 1] == str(ROOT / "baselines/Ayo")
        assert "--variants" not in ayo  # Complete Ayo never uses the HeraSys dispatcher.
        assert hers[hers.index("--output") + 1] == str(tmp_path / "HeraSys")
        assert ayo[ayo.index("--output") + 1] == str(tmp_path / "Ayo")


def test_snapshot_extraction_rejects_path_traversal_and_links(tmp_path):
    for name, kind in (("../escape.py", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
        snapshot = tmp_path / "source.tar.gz"
        with tarfile.open(snapshot, "w:gz") as archive:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "../../escape"
            archive.addfile(member, io.BytesIO(b""))
        with pytest.raises(ValueError, match="Unsafe archive member"):
            extract_snapshot(snapshot, tmp_path / "checkout")
    assert not (tmp_path / "escape.py").exists()


def test_distribution_includes_all_herasys_source_packages():
    packages = find_packages(where=str(ROOT), include=["HeraSys", "HeraSys.*"])
    assert set(packages) == {
        "HeraSys", "HeraSys.configs", "HeraSys.dags", "HeraSys.engines",
        "HeraSys.modules", "HeraSys.opt_pass", "HeraSys.queries",
        "HeraSys.schedulers", "HeraSys.vis",
    }


def test_source_snapshot_contains_the_complete_current_package(tmp_path):
    (tmp_path / "HeraSys").mkdir()
    save_source_snapshot({}, tmp_path, ROOT, "current")
    with tarfile.open(tmp_path / "HeraSys/source.tar.gz") as archive:
        for path in (ROOT / "HeraSys").rglob("*.py"):
            assert archive.extractfile(str(path.relative_to(ROOT))).read() == path.read_bytes()
        assert not any("__pycache__" in p for p in archive.getnames())


def test_measured_source_snapshot_is_preserved_byte_for_byte(tmp_path):
    snapshot = tmp_path / "original.tar.gz"
    snapshot.write_bytes(b"measured-source-snapshot")
    output = tmp_path / "run"
    (output / "HeraSys").mkdir(parents=True)
    config = {"herasys": {"snapshot": "original.tar.gz"}}
    save_source_snapshot(config, output, ROOT, "measured", root=tmp_path)
    assert (output / "HeraSys/source.tar.gz").read_bytes() == snapshot.read_bytes()


def test_occupied_gpus_are_not_used_or_stopped(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []

    def check_output(command, **kwargs):
        calls.append(command)
        return "0\n1\n2\n3\n" if "--query-gpu=index" in command else "12345\n"

    monkeypatch.setattr("scripts.reproduce.subprocess.check_output", check_output)
    with pytest.raises(RuntimeError, match="GPUs are occupied"):
        require_idle_gpus()
    assert len(calls) == 2
    assert all(command[0] == "nvidia-smi" for command in calls)
