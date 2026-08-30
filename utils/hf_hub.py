"""HuggingFace Hub helpers — token resolution and authenticated downloads.

Two things live here so the rest of the codebase does not repeat them:

* :func:`get_hf_token` — the token StemForge should use, resolved the same
  way everywhere (the token saved through the UI wins, then the standard
  ``huggingface_hub`` sources: ``HF_TOKEN`` and ``huggingface-cli login``).
* :func:`download_file` — fetch a URL to a local path, routing anything on
  huggingface.co through ``hf_hub_download`` so the request carries that
  token. A plain ``urllib`` GET is anonymous, which the Hub answers with
  HTTP 401 for any repo that is gated or rate-limits anonymous traffic.

Non-Hub URLs (GitHub releases, raw.githubusercontent.com) fall through to a
plain download, since there is nothing to authenticate with.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import urllib.request
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

log = logging.getLogger("stemforge.utils.hf_hub")

#: Hostnames that serve the HuggingFace Hub.
_HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})

#: URL path prefix -> ``hf_hub_download`` repo_type. Models carry no prefix.
_REPO_TYPE_PREFIXES = {"datasets": "dataset", "spaces": "space"}


@dataclass(frozen=True, slots=True)
class HFFileRef:
    """A file on the Hub, as named by ``hf_hub_download``."""

    repo_id: str
    filename: str
    revision: str
    repo_type: str


def get_hf_token() -> str | None:
    """Return the HuggingFace token to authenticate with, or ``None``.

    Priority: the token saved through StemForge's UI (``POST /api/hf-token``),
    then whatever ``huggingface_hub`` finds — the ``HF_TOKEN`` environment
    variable or a ``huggingface-cli login`` credential.
    """
    try:
        from utils.platform import get_data_dir

        stored = get_data_dir() / "hf_token"
        if stored.is_file():
            token = stored.read_text().strip()
            if token:
                return token
    except Exception:  # unreadable token file must never break a download
        log.debug("Could not read the StemForge HuggingFace token", exc_info=True)

    try:
        from huggingface_hub import get_token  # type: ignore[import]

        return get_token()
    except Exception:
        log.debug("huggingface_hub could not supply a token", exc_info=True)
        return None


def parse_hf_url(url: str) -> HFFileRef | None:
    """Parse a Hub ``.../resolve/...`` download URL, or return ``None``.

    ``None`` means "not a Hub file URL" — the caller should download it
    directly. Handles the ``datasets/`` and ``spaces/`` prefixes, single- and
    two-segment repo ids, and percent-encoded revisions such as
    ``refs%2Fpr%2F1``.
    """
    parts = urlsplit(url)
    if parts.netloc.lower() not in _HF_HOSTS:
        return None

    segments = [s for s in parts.path.split("/") if s]

    repo_type = "model"
    if segments and segments[0] in _REPO_TYPE_PREFIXES:
        repo_type = _REPO_TYPE_PREFIXES[segments[0]]
        segments = segments[1:]

    if "resolve" not in segments:
        return None
    marker = segments.index("resolve")

    # Need at least one repo segment before the marker and revision + filename
    # after it.
    if marker < 1 or len(segments) < marker + 3:
        return None

    return HFFileRef(
        repo_id="/".join(segments[:marker]),
        filename="/".join(unquote(s) for s in segments[marker + 2:]),
        revision=unquote(segments[marker + 1]),
        repo_type=repo_type,
    )


def download_file(url: str, dest: pathlib.Path) -> None:
    """Download *url* to *dest*, authenticating when it is a Hub URL.

    The write is atomic: *dest* only appears once the bytes are complete, so
    an interrupted download cannot leave a truncated file that later looks
    cached.

    Raises
    ------
    RuntimeError
        On a Hub authentication or access failure, with the steps to fix it.
    Exception
        Whatever the underlying transport raised, otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    ref = parse_hf_url(url)
    if ref is not None:
        _download_from_hub(ref, dest)
    else:
        _download_direct(url, dest)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _download_direct(url: str, dest: pathlib.Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _download_from_hub(ref: HFFileRef, dest: pathlib.Path) -> None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        log.warning("huggingface_hub is unavailable; falling back to an anonymous download")
        _download_direct(
            f"https://huggingface.co/{ref.repo_id}/resolve/{ref.revision}/{ref.filename}",
            dest,
        )
        return

    # Download into a scratch directory beside dest rather than the shared HF
    # cache: the file is moved into our own cache afterwards, so a cache copy
    # would just be a second gigabyte of the same checkpoint.
    staging = dest.parent / f".{dest.name}.hfdl"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        fetched = hf_hub_download(
            repo_id=ref.repo_id,
            filename=ref.filename,
            revision=ref.revision,
            repo_type=ref.repo_type,
            token=get_hf_token(),
            local_dir=str(staging),
        )
        _move_into_place(pathlib.Path(fetched), dest)
    except Exception as exc:
        raise _translate_hub_error(exc, ref) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _move_into_place(fetched: pathlib.Path, dest: pathlib.Path) -> None:
    """Move *fetched* to *dest*, materialising it if it is a symlink.

    huggingface_hub before 0.26 could satisfy ``local_dir`` with a symlink into
    the shared Hub cache. Moving the link would leave *dest* dangling as soon
    as that cache is pruned, so copy the bytes in that case.
    """
    if fetched.is_symlink():
        shutil.copyfile(fetched.resolve(), dest)
        return
    try:
        fetched.replace(dest)
    except OSError:  # staging and dest on different filesystems
        shutil.move(str(fetched), str(dest))


def _translate_hub_error(exc: Exception, ref: HFFileRef) -> Exception:
    """Turn an auth/access failure into a message that says what to do."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    unauthorized = (
        "401" in text
        or "403" in text
        or "gated" in lowered
        or "unauthorized" in lowered
        or "authenticated" in lowered
        or "access to model" in lowered
    )
    if not unauthorized:
        return exc

    url = f"https://huggingface.co/{ref.repo_id}"
    have_token = get_hf_token() is not None
    if have_token:
        reason = (
            f"The configured HuggingFace token was rejected for {ref.repo_id}.\n"
            f"  1. Visit {url} and accept the model's terms with the same account.\n"
            "  2. Check the token still exists and has read access."
        )
    else:
        reason = (
            f"{ref.repo_id} needs a HuggingFace account to download.\n"
            f"  1. Visit {url} and accept the model's terms (a free account).\n"
            "  2. Add a token in the Synth tab's HuggingFace banner, run\n"
            "     'huggingface-cli login', or set the HF_TOKEN environment variable."
        )
    return RuntimeError(f"{reason}\n\nUnderlying error: {text}")
