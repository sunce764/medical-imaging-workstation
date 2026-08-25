"""Machine-readable provenance contract for performance experiments.

This module records observations; it does not run an experiment or alter an
existing result.  Keep it stdlib-only so benchmark metadata can still be
written when optional analysis packages are unavailable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str) -> str:
    """Return the SHA-256 of one exact artifact without loading it into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(project_root: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = result.stdout.strip()
    return head if re.fullmatch(r"[0-9a-f]{40}", head) else None


def _total_memory_gib() -> float | None:
    """Best-effort physical RAM capacity; absence must not block a run."""
    try:
        if sys.platform == "darwin":
            raw = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
            ).strip()
            return round(int(raw) / (1024 ** 3), 3)
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(int(pages) * int(page_size) / (1024 ** 3), 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def machine_snapshot() -> dict[str, Any]:
    """Capture stable-enough machine fields needed to interpret a timing."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_gib": _total_memory_gib(),
        "python_version": platform.python_version(),
    }


def process_peak_memory_gib() -> tuple[float, str]:
    """Return process-lifetime ru_maxrss in GiB and its measurement semantics."""
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        gib = rss / (1024 ** 3)  # macOS reports bytes
        raw_unit = "bytes"
    else:
        gib = rss / (1024 ** 2)  # Linux reports KiB
        raw_unit = "KiB"
    method = (
        "resource.getrusage(RUSAGE_SELF).ru_maxrss; process-lifetime high-water "
        f"mark; raw unit={raw_unit}"
    )
    return float(gib), method


def _software_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _model_record(model_files: Iterable[str]) -> dict[str, Any]:
    paths = [os.path.abspath(os.fspath(p)) for p in model_files]
    if not paths:
        raise ValueError("model_files must contain at least one file")
    if len({os.path.basename(p) for p in paths}) != len(paths):
        raise ValueError("model_files must have unique basenames")

    files = []
    for path in sorted(paths, key=os.path.basename):
        if not os.path.isfile(path):
            raise ValueError(f"model file does not exist: {path}")
        files.append({
            # Avoid leaking a user's absolute filesystem path into public artifacts.
            "name": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        })
    combined = hashlib.sha256()
    for item in files:
        combined.update(f"{item['name']}\0{item['sha256']}\n".encode())
    return {"files": files, "combined_sha256": combined.hexdigest()}


def build_performance_artifact(
    *,
    script: str,
    mode: str,
    project_root: str,
    model_files: Iterable[str],
    configuration: Mapping[str, Any],
    wall_time_seconds: float,
    peak_memory_gib: float,
    peak_memory_method: str,
    dependency_packages: Iterable[str] = (),
    extra_measurements: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one performance-run provenance document."""
    measurements: dict[str, Any] = {
        "wall_time_seconds": float(wall_time_seconds),
        "peak_memory_gib": float(peak_memory_gib),
        "peak_memory_method": peak_memory_method,
    }
    if extra_measurements:
        overlap = set(measurements) & set(extra_measurements)
        if overlap:
            raise ValueError(f"extra_measurements overwrite required fields: {sorted(overlap)}")
        measurements.update(dict(extra_measurements))

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z"),
        "producer": {
            "script": script,
            "mode": mode,
            "git_head": _git_head(project_root),
        },
        "machine": machine_snapshot(),
        "software": _software_versions(dependency_packages),
        "configuration": dict(configuration),
        "measurements": measurements,
        "model": _model_record(model_files),
    }
    return validate_performance_artifact(artifact)


def validate_performance_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete or internally inconsistent performance provenance."""
    required = {
        "schema_version", "created_at_utc", "producer", "machine", "software",
        "configuration", "measurements", "model",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"missing artifact fields: {sorted(missing)}")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if not isinstance(artifact["configuration"], dict):
        raise ValueError("configuration must be an object")

    machine = artifact["machine"]
    machine_required = {"system", "release", "machine", "logical_cpu_count",
                        "physical_memory_gib", "python_version"}
    if not isinstance(machine, dict) or not machine_required <= set(machine):
        raise ValueError("machine snapshot is incomplete")

    measurements = artifact["measurements"]
    for key in ("wall_time_seconds", "peak_memory_gib"):
        value = measurements.get(key) if isinstance(measurements, dict) else None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be a finite non-negative number")
    if not measurements.get("peak_memory_method"):
        raise ValueError("peak_memory_method is required")

    model = artifact["model"]
    files = model.get("files") if isinstance(model, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("model.files must be a non-empty list")
    names = []
    combined = hashlib.sha256()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("each model file must be an object")
        name, size, digest = item.get("name"), item.get("size_bytes"), item.get("sha256")
        if not isinstance(name, str) or not name or os.path.basename(name) != name:
            raise ValueError("model file names must be non-empty basenames")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("model file size_bytes must be a non-negative integer")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ValueError("model file sha256 is invalid")
        names.append(name)
        combined.update(f"{name}\0{digest}\n".encode())
    if len(names) != len(set(names)):
        raise ValueError("model file names must be unique")
    if model.get("combined_sha256") != combined.hexdigest():
        raise ValueError("model combined_sha256 is inconsistent with its files")

    # Fail here, rather than after a long benchmark, if callers supplied values
    # that cannot be represented in a portable JSON artifact.
    try:
        json.dumps(artifact, allow_nan=False)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"artifact is not strict JSON: {ex}") from ex
    return artifact


def artifact_timestamp_slug(artifact: Mapping[str, Any]) -> str:
    """Filesystem-safe, microsecond-resolution timestamp for append-only runs."""
    value = str(artifact["created_at_utc"])
    return re.sub(r"[^0-9TZ]", "", value)


def write_performance_artifact(path: str, artifact: dict[str, Any]) -> None:
    """Validate then atomically publish one JSON artifact."""
    validate_performance_artifact(artifact)
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".performance-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
