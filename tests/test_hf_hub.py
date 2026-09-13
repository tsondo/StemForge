"""Tests for utils.hf_hub — Hub URL routing, token resolution, error text.

The bug this guards: BS-Roformer fetched its checkpoints with a plain
``urllib`` GET, which is anonymous, so huggingface.co answered a gated repo
with HTTP 401 and separation failed with an opaque "Download failed".
Hub URLs must now be recognised and routed through ``hf_hub_download``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# ── root on sys.path so project imports resolve ───────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import hf_hub  # noqa: E402


# ── URL parsing ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "url, repo_id, filename, revision, repo_type",
    [
        (
            "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt",
            "jarredou/BS-ROFO-SW-Fixed",
            "BS-Rofo-SW-Fixed.ckpt",
            "main",
            "model",
        ),
        # Nested file paths keep their directories.
        (
            "https://huggingface.co/owner/repo/resolve/main/sub/dir/model.ckpt",
            "owner/repo",
            "sub/dir/model.ckpt",
            "main",
            "model",
        ),
        # Canonical single-segment repo ids.
        (
            "https://huggingface.co/gpt2/resolve/main/config.json",
            "gpt2",
            "config.json",
            "main",
            "model",
        ),
        # Percent-encoded revisions (a PR ref).
        (
            "https://huggingface.co/owner/repo/resolve/refs%2Fpr%2F1/f.bin",
            "owner/repo",
            "f.bin",
            "refs/pr/1",
            "model",
        ),
        # Dataset and space prefixes select the repo type.
        (
            "https://huggingface.co/datasets/owner/repo/resolve/main/f.wav",
            "owner/repo",
            "f.wav",
            "main",
            "dataset",
        ),
        # The short hf.co host is the same Hub.
        (
            "https://hf.co/owner/repo/resolve/v1.0/f.ckpt",
            "owner/repo",
            "f.ckpt",
            "v1.0",
            "model",
        ),
    ],
)
def test_parse_hf_url(url, repo_id, filename, revision, repo_type):
    ref = hf_hub.parse_hf_url(url)
    assert ref is not None
    assert (ref.repo_id, ref.filename, ref.revision, ref.repo_type) == (
        repo_id,
        filename,
        revision,
        repo_type,
    )


@pytest.mark.parametrize(
    "url",
    [
        # The two non-Hub hosts the Roformer registry actually uses.
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/m.ckpt",
        "https://raw.githubusercontent.com/TRvlvr/application_data/main/c.yaml",
        # A Hub page, not a file download.
        "https://huggingface.co/owner/repo",
        # A look-alike host must not be handed our token.
        "https://huggingface.co.evil.example/owner/repo/resolve/main/f.ckpt",
    ],
)
def test_non_hub_urls_are_not_parsed(url):
    assert hf_hub.parse_hf_url(url) is None


# ── Routing ───────────────────────────────────────────────────────────────

def test_hub_url_goes_through_hf_hub_download(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        staged = pathlib.Path(kwargs["local_dir"]) / kwargs["filename"]
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"weights")
        return str(staged)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        type(sys)("huggingface_hub"),
    )
    sys.modules["huggingface_hub"].hf_hub_download = fake_hf_hub_download
    monkeypatch.setattr(hf_hub, "get_hf_token", lambda: "hf_testtoken")

    dest = tmp_path / "roformer-jarredou-6stem.ckpt"
    hf_hub.download_file(
        "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt",
        dest,
    )

    assert dest.read_bytes() == b"weights"
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "jarredou/BS-ROFO-SW-Fixed"
    assert calls[0]["token"] == "hf_testtoken", "the token must reach the Hub request"
    # The staging directory must not survive the download.
    assert not list(tmp_path.glob(".*.hfdl"))


def test_non_hub_url_uses_a_plain_download(tmp_path, monkeypatch):
    urls: list[str] = []

    def fake_urlretrieve(url, filename):
        urls.append(url)
        pathlib.Path(filename).write_bytes(b"ckpt")

    monkeypatch.setattr(hf_hub.urllib.request, "urlretrieve", fake_urlretrieve)

    dest = tmp_path / "viperx.ckpt"
    url = "https://github.com/TRvlvr/model_repo/releases/download/all/m.ckpt"
    hf_hub.download_file(url, dest)

    assert urls == [url]
    assert dest.read_bytes() == b"ckpt"


def test_failed_direct_download_leaves_no_partial_file(tmp_path, monkeypatch):
    def failing_urlretrieve(url, filename):
        pathlib.Path(filename).write_bytes(b"half a checkpo")
        raise OSError("connection reset")

    monkeypatch.setattr(hf_hub.urllib.request, "urlretrieve", failing_urlretrieve)

    dest = tmp_path / "m.ckpt"
    with pytest.raises(OSError):
        hf_hub.download_file("https://example.com/m.ckpt", dest)

    assert not dest.exists(), "a truncated file would look like a valid cache hit"
    assert list(tmp_path.iterdir()) == []


# ── Error translation ─────────────────────────────────────────────────────

def _fail_hub_download(monkeypatch, exc: Exception) -> None:
    module = type(sys)("huggingface_hub")

    def fake(**kwargs):
        raise exc

    module.hf_hub_download = fake
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_401_without_a_token_says_how_to_add_one(tmp_path, monkeypatch):
    _fail_hub_download(monkeypatch, OSError("401 Client Error: Unauthorized"))
    monkeypatch.setattr(hf_hub, "get_hf_token", lambda: None)

    with pytest.raises(RuntimeError) as excinfo:
        hf_hub.download_file(
            "https://huggingface.co/owner/repo/resolve/main/m.ckpt", tmp_path / "m.ckpt"
        )

    message = str(excinfo.value)
    assert "https://huggingface.co/owner/repo" in message
    assert "HF_TOKEN" in message


def test_401_with_a_token_says_the_token_was_rejected(tmp_path, monkeypatch):
    _fail_hub_download(monkeypatch, OSError("403 Client Error: gated repo"))
    monkeypatch.setattr(hf_hub, "get_hf_token", lambda: "hf_testtoken")

    with pytest.raises(RuntimeError) as excinfo:
        hf_hub.download_file(
            "https://huggingface.co/owner/repo/resolve/main/m.ckpt", tmp_path / "m.ckpt"
        )

    assert "was rejected" in str(excinfo.value)


def test_unrelated_hub_failures_are_not_reworded(tmp_path, monkeypatch):
    _fail_hub_download(monkeypatch, TimeoutError("read timed out"))
    monkeypatch.setattr(hf_hub, "get_hf_token", lambda: None)

    with pytest.raises(TimeoutError):
        hf_hub.download_file(
            "https://huggingface.co/owner/repo/resolve/main/m.ckpt", tmp_path / "m.ckpt"
        )


# ── Token resolution ──────────────────────────────────────────────────────

def test_stored_token_wins_over_huggingface_cli(tmp_path, monkeypatch):
    (tmp_path / "hf_token").write_text("  hf_stored  \n")

    platform_module = type(sys)("utils.platform")
    platform_module.get_data_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "utils.platform", platform_module)

    hub_module = type(sys)("huggingface_hub")
    hub_module.get_token = lambda: "hf_from_cli"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    assert hf_hub.get_hf_token() == "hf_stored"


def test_empty_stored_token_falls_through_to_huggingface_cli(tmp_path, monkeypatch):
    (tmp_path / "hf_token").write_text("   \n")

    platform_module = type(sys)("utils.platform")
    platform_module.get_data_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "utils.platform", platform_module)

    hub_module = type(sys)("huggingface_hub")
    hub_module.get_token = lambda: "hf_from_cli"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    assert hf_hub.get_hf_token() == "hf_from_cli"


def test_no_token_anywhere_is_none_not_an_error(tmp_path, monkeypatch):
    platform_module = type(sys)("utils.platform")
    platform_module.get_data_dir = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "utils.platform", platform_module)

    hub_module = type(sys)("huggingface_hub")
    hub_module.get_token = lambda: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    assert hf_hub.get_hf_token() is None
