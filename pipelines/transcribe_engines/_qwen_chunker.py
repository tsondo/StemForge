"""Overlap-and-stitch chunking for Qwen2-Audio.

Qwen2-Audio has a hard 30-second audio context limit. For longer audio,
we slice the input into 24-second chunks with 6-second overlaps, transcribe
each chunk independently, and stitch the resulting text using longest-common-
substring matching on the overlapping tail/head regions.

Design constants
----------------
CHUNK_DURATION_S      : Audio window passed to the model per call. Must be
                        comfortably below Qwen's 30s limit to leave room for
                        processor padding and prompt tokens.
OVERLAP_DURATION_S    : Redundant audio at each chunk boundary. Chosen to
                        reliably contain 2+ alignable tokens on sparse sung
                        audio while keeping compute overhead acceptable.
MIN_MATCH_TOKENS      : Minimum LCS length (in tokens) to accept a stitch.
                        Below this, fall back to concatenation with no
                        deduplication — redundant text is better than dropped
                        text.
ALIGN_WINDOW_FRACTION : Fraction of each chunk's text searched for an
                        alignment match. 0.30 means the last 30% of chunk N
                        is matched against the first 30% of chunk N+1.
                        Larger than the overlap fraction (20%) to absorb
                        timing jitter — the text-time mapping is not linear.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

CHUNK_DURATION_S: float = 24.0
OVERLAP_DURATION_S: float = 6.0
STEP_DURATION_S: float = CHUNK_DURATION_S - OVERLAP_DURATION_S  # 18.0

MIN_MATCH_TOKENS: int = 2
ALIGN_WINDOW_FRACTION: float = 0.30
# Floor for the fraction-based alignment window.  ALIGN_WINDOW_FRACTION is
# calibrated for realistic 24-second chunks (~50–200 tokens), where 30%
# gives a 15–60 token window — comfortably larger than typical overlaps.
# Short chunks (e.g. an instrumental-heavy intro that produces only a few
# transcribed words) would otherwise get windows too small to fit the
# overlap region, causing the stitcher to fall back to concatenation when
# a clean match was actually available.  Clamped to chunk length below.
MIN_ALIGN_WINDOW: int = 8

# Sample rate used by Qwen2-Audio's audio processor.
SAMPLE_RATE: int = 16_000


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A single audio slice scheduled for transcription."""
    index: int                  # 0-based chunk index
    start_s: float              # start time in source audio
    end_s: float                # end time in source audio
    samples: np.ndarray         # mono int16/float32 audio at SAMPLE_RATE


def slice_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[AudioChunk]:
    """Slice a mono audio array into overlapping chunks.

    The final chunk extends to the end of the audio regardless of size — it
    will be shorter than CHUNK_DURATION_S if the audio doesn't divide evenly.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    total_samples = len(audio)
    chunk_samples = int(CHUNK_DURATION_S * SAMPLE_RATE)
    step_samples = int(STEP_DURATION_S * SAMPLE_RATE)

    # Single-chunk fast path: audio fits inside one window.
    if total_samples <= chunk_samples:
        return [AudioChunk(
            index=0,
            start_s=0.0,
            end_s=total_samples / SAMPLE_RATE,
            samples=audio,
        )]

    chunks: list[AudioChunk] = []
    pos = 0
    idx = 0
    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunks.append(AudioChunk(
            index=idx,
            start_s=pos / SAMPLE_RATE,
            end_s=end / SAMPLE_RATE,
            samples=audio[pos:end],
        ))
        # If this chunk reached the end, stop — don't emit a redundant final
        # chunk that's just the tail of the previous one.
        if end >= total_samples:
            break
        pos += step_samples
        idx += 1
    return chunks


# ── Text stitching ────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\S+")


def _tokenize(text: str) -> list[str]:
    """Whitespace tokens, preserving original casing and punctuation.

    Normalization for *matching* happens in _norm_for_match. We keep the
    original tokens around so the final stitched output preserves the
    model's chosen capitalization and punctuation.
    """
    return _TOKEN_RE.findall(text)


def _norm_for_match(token: str) -> str:
    """Casefold and strip a small set of repetition-prone punctuation."""
    stripped = token.lower()
    for ch in "¡!¿?.,;:\"'":
        stripped = stripped.replace(ch, "")
    return stripped


def _longest_common_substring(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """Find the longest contiguous token substring shared by a and b.

    Returns (length, a_start_idx, b_start_idx). Length 0 means no match.
    Token equality is determined under _norm_for_match.

    O(len(a) * len(b)) time, O(min(len(a), len(b))) space.
    """
    if not a or not b:
        return (0, 0, 0)
    na = [_norm_for_match(t) for t in a]
    nb = [_norm_for_match(t) for t in b]

    # Rolling 1-D DP — only need the previous row.
    prev = [0] * (len(nb) + 1)
    curr = [0] * (len(nb) + 1)
    best_len = 0
    best_a = 0
    best_b = 0
    for i in range(1, len(na) + 1):
        for j in range(1, len(nb) + 1):
            if na[i - 1] == nb[j - 1] and na[i - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_a = i - best_len
                    best_b = j - best_len
            else:
                curr[j] = 0
        prev, curr = curr, prev
        for j in range(len(curr)):
            curr[j] = 0
    return (best_len, best_a, best_b)


def stitch_chunks(chunk_texts: list[str]) -> str:
    """Stitch overlapping chunk transcripts into a single transcript.

    For each adjacent pair (N, N+1):
      1. Take the last ALIGN_WINDOW_FRACTION of N's tokens (tail).
      2. Take the first ALIGN_WINDOW_FRACTION of N+1's tokens (head).
      3. Find the longest common substring between tail and head.
      4. If LCS length >= MIN_MATCH_TOKENS:
           - Cut N at the *start* of its matched region.
           - Cut N+1 at the *end* of its matched region.
           - The matched region itself is kept from N (could be either; N
             tends to be slightly more reliable than the start of N+1, which
             can show splice artifacts).
      5. Otherwise: fall back to plain concatenation, accepting duplication.
         This is the correct degradation — redundant text is recoverable by
         the user; dropped text is not.
    """
    if not chunk_texts:
        return ""
    if len(chunk_texts) == 1:
        return chunk_texts[0].strip()

    # Tokenize all chunks once.
    tokens_per_chunk: list[list[str]] = [_tokenize(t) for t in chunk_texts]

    # Start with chunk 0's tokens in full.
    out_tokens: list[str] = list(tokens_per_chunk[0])

    for i in range(1, len(tokens_per_chunk)):
        next_tokens = tokens_per_chunk[i]
        if not next_tokens:
            continue
        if not out_tokens:
            out_tokens = list(next_tokens)
            continue

        # Tail of accumulated output, head of next chunk.
        tail_len = min(
            len(out_tokens),
            max(MIN_ALIGN_WINDOW, int(len(out_tokens) * ALIGN_WINDOW_FRACTION)),
        )
        head_len = min(
            len(next_tokens),
            max(MIN_ALIGN_WINDOW, int(len(next_tokens) * ALIGN_WINDOW_FRACTION)),
        )
        tail = out_tokens[-tail_len:]
        head = next_tokens[:head_len]

        length, tail_start, head_start = _longest_common_substring(tail, head)

        if length >= MIN_MATCH_TOKENS:
            # Absolute index in out_tokens where the match starts.
            cut_out_at = len(out_tokens) - tail_len + tail_start
            # Absolute index in next_tokens where the match ends.
            cut_next_at = head_start + length
            # Keep out_tokens[:cut_out_at] + tail-match + next_tokens[cut_next_at:]
            matched_region = tail[tail_start:tail_start + length]
            out_tokens = out_tokens[:cut_out_at] + matched_region + list(next_tokens[cut_next_at:])
            log.debug(
                "Stitched chunk %d → %d: matched %d tokens (%r)",
                i - 1, i, length, " ".join(matched_region),
            )
        else:
            # No reliable match — concatenate with a soft separator.
            # Use a newline so the unstitched join is visible in the .txt
            # output without inventing punctuation the model didn't emit.
            out_tokens = out_tokens + ["\n"] + list(next_tokens)
            log.info(
                "No alignment between chunk %d and %d (best match: %d tokens) "
                "— falling back to concatenation.",
                i - 1, i, length,
            )

    # Reassemble with single spaces, but preserve any "\n" sentinel we inserted.
    result_parts: list[str] = []
    for tok in out_tokens:
        if tok == "\n":
            # Strip trailing space from previous part, append newline.
            if result_parts and result_parts[-1].endswith(" "):
                result_parts[-1] = result_parts[-1].rstrip()
            result_parts.append("\n")
        else:
            if result_parts and not result_parts[-1].endswith("\n"):
                result_parts.append(" ")
            result_parts.append(tok)
    return "".join(result_parts).strip()
