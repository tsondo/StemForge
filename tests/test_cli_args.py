"""Tests for run.py's CLI surface — currently the TLS flag pair.

The mismatched-flag rule matters because a user who typed one --ssl-* flag
intends TLS; silently starting over HTTP would reproduce exactly the
insecure-origin confusion the flags exist to remove (see
docs/MIDI_OUT_SECURE_CONTEXT_SPEC.md). The validation predicate is tested
directly rather than through main(), which acquires the GPU lock.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# ── root on sys.path so project imports resolve ───────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run  # noqa: E402


def _parse(argv: list[str]):
    old = sys.argv
    sys.argv = ["stemforge", *argv]
    try:
        return run._parse_args()
    finally:
        sys.argv = old


def test_ssl_flags_default_to_none():
    args = _parse([])
    assert args.ssl_certfile is None
    assert args.ssl_keyfile is None


def test_ssl_flags_are_accepted_and_stored():
    args = _parse(["--ssl-certfile", "cert.pem", "--ssl-keyfile", "key.pem"])
    assert args.ssl_certfile == "cert.pem"
    assert args.ssl_keyfile == "key.pem"


@pytest.mark.parametrize(
    "argv",
    [
        ["--ssl-certfile", "cert.pem"],
        ["--ssl-keyfile", "key.pem"],
    ],
)
def test_a_lone_ssl_flag_fails_the_pair_check(argv):
    """One flag without the other must refuse to start, not fall back to HTTP."""
    args = _parse(argv)
    # The exact predicate main() checks before acquiring the GPU lock.
    assert bool(args.ssl_certfile) != bool(args.ssl_keyfile)


def test_paired_and_absent_flags_pass_the_pair_check():
    for argv in ([], ["--ssl-certfile", "c.pem", "--ssl-keyfile", "k.pem"]):
        args = _parse(argv)
        assert bool(args.ssl_certfile) == bool(args.ssl_keyfile)
