"""Guard: the platform pyproject variants must declare the same dependencies.

``pyproject.toml`` (CUDA) is the working file; ``pyproject.toml.ROCM`` and
``pyproject.toml.MAC`` are copy-over-the-top variants that swap the torch
wheel index and little else. Because they are never touched during normal
development, every dependency added to the CUDA file silently fails to reach
them — which is how ``adtof-pytorch`` came to be missing from the ROCm file,
breaking ADTOF drum transcription for AMD users, and how the macOS file drifted
a whole UI behind.

This test fails the moment a dependency is not declared by every variant,
unless ``PLATFORM_EXCEPTIONS`` records which variants may omit it and why.
Exceptions are exact: a package listed as omitted from a variant must in fact
be absent there, so a stale exemption cannot quietly re-open the hole.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Every platform variant. All of them must declare every dependency, except
#: as recorded in PLATFORM_EXCEPTIONS below.
VARIANTS: tuple[str, ...] = (
    "pyproject.toml",
    "pyproject.toml.ROCM",
    "pyproject.toml.MAC",
)

#: Requirement name -> (variants that must NOT declare it, why).
PLATFORM_EXCEPTIONS: dict[str, tuple[frozenset[str], str]] = {
    "nvidia-cuda-runtime-cu12": (
        frozenset({"pyproject.toml.ROCM", "pyproject.toml.MAC"}),
        "CUDA runtime shim for nano-vllm's flash-attn; nothing to shim elsewhere",
    ),
    "triton-rocm": (
        frozenset({"pyproject.toml", "pyproject.toml.MAC"}),
        "ROCm build of triton, published only on the rocm wheel index",
    ),
    "nano-vllm": (
        frozenset({"pyproject.toml.MAC"}),
        "ACE-Step excludes it on darwin/arm64 (MLX path); its flash-attn "
        "requirements are CUDA wheel URLs",
    ),
}


def _canonical(name: str) -> str:
    """PEP 503 canonical form, so ``pretty_midi`` and ``pretty-midi`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(variant: str) -> dict[str, str]:
    """Map canonical requirement name -> the full requirement string."""
    data = tomllib.loads((ROOT / variant).read_text(encoding="utf-8"))
    reqs: dict[str, str] = {}
    for raw in data["project"]["dependencies"]:
        spec = raw.strip()
        # The name runs up to the first extras/version/url/marker delimiter.
        name = re.split(r"[\s\[<>=!~@;]", spec, maxsplit=1)[0]
        reqs[_canonical(name)] = spec
    return reqs


def _all_requirements() -> dict[str, dict[str, str]]:
    return {variant: _requirements(variant) for variant in VARIANTS}


def _omitted_from(name: str) -> frozenset[str]:
    exception = PLATFORM_EXCEPTIONS.get(name)
    return exception[0] if exception else frozenset()


def test_every_variant_declares_every_dependency():
    reqs = _all_requirements()
    everything = {name for variant in reqs.values() for name in variant}

    problems: list[str] = []
    for name in sorted(everything):
        omitted = _omitted_from(name)
        expected = {v for v in VARIANTS if v not in omitted}
        declared = {v for v in VARIANTS if name in reqs[v]}

        for variant in sorted(expected - declared):
            problems.append(f"  {name!r} is missing from {variant}")
        for variant in sorted(declared - expected):
            problems.append(
                f"  {name!r} is declared in {variant}, which PLATFORM_EXCEPTIONS "
                f"says must omit it — drop the stale exemption"
            )

    assert not problems, (
        "The platform pyproject variants have drifted:\n"
        + "\n".join(problems)
        + "\n\nAdd the dependency to the other variants, or — if it genuinely "
        "belongs to some platforms only — record it in PLATFORM_EXCEPTIONS in "
        "tests/test_pyproject_variants.py with the reason."
    )


def test_shared_dependencies_have_matching_specifiers():
    """A version pin bumped in one variant but not the others is drift too."""
    reqs = _all_requirements()
    base, *others = VARIANTS

    mismatched: list[str] = []
    for name, spec in sorted(reqs[base].items()):
        for variant in others:
            if name in reqs[variant] and reqs[variant][name] != spec:
                mismatched.append(
                    f"  {name}:\n    {base:<20} {spec}\n"
                    f"    {variant:<20} {reqs[variant][name]}"
                )

    assert not mismatched, (
        "Shared dependencies are pinned differently across variants:\n"
        + "\n".join(mismatched)
    )


def test_platform_exceptions_name_real_variants():
    """A typo'd filename in PLATFORM_EXCEPTIONS would silently exempt nothing."""
    for name, (omitted, _reason) in PLATFORM_EXCEPTIONS.items():
        unknown = omitted - set(VARIANTS)
        assert not unknown, f"{name!r} exempts unknown variants: {sorted(unknown)}"
        assert omitted != set(VARIANTS), (
            f"{name!r} is exempted from every variant — then nothing declares it, "
            f"and the entry should just be deleted."
        )
