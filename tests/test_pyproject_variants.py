"""Guard: the platform pyproject variants must declare the same dependencies.

``pyproject.toml`` (CUDA) is the working file; ``pyproject.toml.ROCM`` is a
copy-over-the-top variant that only swaps the torch wheel index. Because the
ROCm file is never touched during normal development, every dependency added
to the CUDA file silently fails to reach it — which is how ``adtof-pytorch``
came to be missing there, breaking ADTOF drum transcription for AMD users.

This test fails the moment the two dependency lists disagree, in either
direction, unless the difference is listed in ``PLATFORM_ONLY`` below as a
deliberate platform split.

``pyproject.toml.MAC`` is deliberately NOT checked: it predates the web UI
(it still declares dearpygui and has no fastapi/uvicorn) and would need a
full rewrite before a diff against it would mean anything.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUDA = ROOT / "pyproject.toml"
ROCM = ROOT / "pyproject.toml.ROCM"

#: Requirement names each variant is expected to declare alone, with the reason.
PLATFORM_ONLY: dict[str, dict[str, str]] = {
    "pyproject.toml": {
        "nvidia-cuda-runtime-cu12": "CUDA runtime shim; no AMD equivalent",
    },
    "pyproject.toml.ROCM": {
        "triton-rocm": "ROCm build of triton, published only on the rocm index",
    },
}


def _canonical(name: str) -> str:
    """PEP 503 canonical form, so ``pretty_midi`` and ``pretty-midi`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(path: Path) -> dict[str, str]:
    """Map canonical requirement name -> the full requirement string."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    reqs: dict[str, str] = {}
    for raw in data["project"]["dependencies"]:
        spec = raw.strip()
        # Name runs up to the first extras/version/url/marker delimiter.
        name = re.split(r"[\s\[<>=!~@;]", spec, maxsplit=1)[0]
        reqs[_canonical(name)] = spec
    return reqs


def _expected_only(path: Path) -> set[str]:
    return {_canonical(n) for n in PLATFORM_ONLY.get(path.name, {})}


def test_no_dependency_missing_from_a_variant():
    cuda, rocm = _requirements(CUDA), _requirements(ROCM)

    missing_from_rocm = (set(cuda) - set(rocm)) - _expected_only(CUDA)
    missing_from_cuda = (set(rocm) - set(cuda)) - _expected_only(ROCM)

    problems = [
        f"  {name!r} is in pyproject.toml but not pyproject.toml.ROCM"
        for name in sorted(missing_from_rocm)
    ] + [
        f"  {name!r} is in pyproject.toml.ROCM but not pyproject.toml"
        for name in sorted(missing_from_cuda)
    ]
    assert not problems, (
        "The platform pyproject variants have drifted:\n"
        + "\n".join(problems)
        + "\n\nAdd the dependency to the other variant, or — if it genuinely "
        "belongs to one platform only — record it in PLATFORM_ONLY in "
        "tests/test_pyproject_variants.py with the reason."
    )


def test_shared_dependencies_have_matching_specifiers():
    """A version pin bumped in one variant but not the other is drift too."""
    cuda, rocm = _requirements(CUDA), _requirements(ROCM)

    mismatched = [
        f"  {name}:\n    pyproject.toml      {cuda[name]}\n"
        f"    pyproject.toml.ROCM {rocm[name]}"
        for name in sorted(set(cuda) & set(rocm))
        if cuda[name] != rocm[name]
    ]
    assert not mismatched, (
        "Shared dependencies are pinned differently across variants:\n"
        + "\n".join(mismatched)
    )


def test_platform_only_entries_are_actually_present():
    """Keep PLATFORM_ONLY honest — a stale exemption hides real drift."""
    for path in (CUDA, ROCM):
        declared = set(_requirements(path))
        for name in _expected_only(path):
            assert name in declared, (
                f"{name!r} is exempted for {path.name} in PLATFORM_ONLY but is "
                f"no longer declared there — drop the stale exemption."
            )
