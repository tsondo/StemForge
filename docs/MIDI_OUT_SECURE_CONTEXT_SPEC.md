# MIDI Out — Secure Context / TLS Support — Technical Specification

**Parent doc:** `WEB_MIDI_OUT_SPEC.md`
**Status:** Ready for implementation
**Scope:** Add optional TLS to the StemForge server so MIDI Out works when the browser is not on `localhost`. Correct one misleading frontend string. Document the three deployment paths. No changes to the MIDI scheduler, the `/api/midi/events` endpoint, the frontend MIDI logic, or any pipeline.

---

## 1 · Motivation

Web MIDI is gated behind `window.isSecureContext`. This is a browser guarantee — a page cannot opt out of it, and no amount of frontend work will change the outcome. The origin must be either HTTPS or a loopback address.

StemForge binds to `0.0.0.0`, so a user on a second machine naturally reaches it at `http://192.168.x.x:8765` or `http://<hostname>:8765`. Neither is a secure context. The MIDI Out panel then hides itself behind an error banner, and the user has no path forward except moving to the server machine.

This was hit in real hardware validation on 2026-08-30 (first AMD/ROCm test session). The user reached StemForge over a non-loopback origin, and the resulting failure looked like a browser permission bug: Brave's per-site MIDI dropdown displayed "Block" and refused to change, because the permission is simply unavailable on an insecure origin. Roughly twenty minutes went into diagnosing Brave's settings before the origin turned out to be the cause. **The current banner text actively contributed to that misdiagnosis** — see §3.2.

Two things are explicitly *not* problems and must not be "fixed":

- **HF Spaces deployment.** Spaces serves over HTTPS, so MIDI Out already works there. This spec is about LAN access only.
- **The `isSecureContext` guard itself.** It is correct. Do not weaken, bypass, or feature-detect around it.

---

## 2 · Design

### 2.1 · Why server-side TLS rather than anything else

Three mechanisms can give the browser a secure context. Only one is a code change:

| Path | Cert handling | Client burden | Code change |
|---|---|---|---|
| Tailscale serve | Auto, publicly trusted | Install Tailscale | None |
| SSH tunnel | None — origin *is* localhost | Terminal access | None |
| StemForge TLS flags | User supplies cert | Trust the CA | **This spec** |

The first two need no repo change and will be documented, not built. The third is worth adding because it is ~15 lines, composes with any cert source (mkcert, an internal CA, a reverse proxy's cert), and is the only option that works without a third-party dependency.

**The scope is deliberately narrow: StemForge accepts a cert and key. It does not generate, manage, renew, or validate them.** Certificate generation is a documentation concern.

### 2.2 · Flag design

Two flags, both optional, mirroring uvicorn's own parameter names so there is nothing new to learn:

```
--ssl-certfile PATH
--ssl-keyfile PATH
```

Supplying one without the other is a startup error, not a silent fallback to HTTP. A user who typed one flag intends TLS; starting insecurely and printing `http://` would reproduce exactly the confusion this spec exists to remove.

No `--ssl-*` environment variable equivalents. The existing env-var surface (`STEMFORGE_PORT`, `MODEL_LOCATION`, etc.) exists for values that vary per deployment; cert paths are already per deployment and the flags are explicit. Adding env vars here widens the surface for no gain.

### 2.3 · Banner behavior

`_print_banner` currently hardcodes `http://localhost:{port}`. Under TLS it must print `https://`. The hostname stays `localhost` — the banner describes how to reach the server from the machine it is running on, which is still correct.

---

## 3 · Implementation

### 3.1 · `run.py`

**Add to `_parse_args`**, after the `--model-dir` argument:

```python
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="Path to a TLS certificate file. Serves over HTTPS. "
             "Required for Web MIDI Out when browsing from another machine.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="Path to the TLS private key file. Must accompany --ssl-certfile.",
    )
```

**Add validation in `main()`**, immediately after `args = _parse_args()` and *before* the GPU lock is acquired — a config error must not leave a lock file behind:

```python
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        print(
            "\n  ERROR: --ssl-certfile and --ssl-keyfile must be given together.\n"
            "  TLS is required for Web MIDI Out from a non-localhost browser.\n"
            "  See the MIDI Out section of README.md for certificate options.\n"
        )
        sys.exit(1)
```

**Change `_print_banner`'s signature** to accept the scheme:

```python
def _print_banner(
    port: int,
    acestep_port: int,
    compose_mode: str,
    gpu: str | None,
    model_dir: str,
    compose_url: str | None = None,
    scheme: str = "http",
) -> None:
```

and its server line (currently line 181):

```python
    print(f"  Server:     {scheme}://localhost:{port}")
```

**Update the call site** (currently line 245):

```python
    _print_banner(
        args.port, args.acestep_port, compose_mode_str,
        args.gpu, str(model_base), args.compose_url,
        scheme="https" if args.ssl_certfile else "http",
    )
```

**Update the `uvicorn.run` call** (currently line 281):

```python
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
```

Passing `None` for both is uvicorn's default and leaves plain HTTP behavior byte-identical. No conditional branch is needed.

### 3.2 · `frontend/components/midi.js` — line 675

The current string is actively misleading:

```js
setBanner('banner-error', 'Web MIDI requires HTTPS or localhost. Open StemForge at http://localhost:8765.');
```

Two faults. It hardcodes port 8765, which is wrong for anyone using `--port`. And it names only the loopback remedy, so a user browsing from a second machine reads an instruction that is impossible to follow and concludes the fault lies elsewhere — which is what happened in the session that prompted this spec.

Replace with:

```js
setBanner(
  'banner-error',
  'Web MIDI requires a secure context. Browse to this server at '
  + `http://localhost:${window.location.port || 80} on the machine running `
  + 'StemForge, or serve it over HTTPS — see the MIDI Out section of README.md.',
);
```

This is the only frontend change in this spec.

### 3.3 · No other code changes

Verified during spec preparation: the frontend uses relative URLs throughout, and there are no `ws://`, `WebSocket`, or `EventSource` usages anywhere in `frontend/`. The `http://` on line 675 is the sole hardcoded protocol in the frontend. Serving over TLS therefore requires no other frontend work and cannot produce mixed-content failures.

---

## 4 · Documentation

### 4.1 · README — new subsection under the MIDI tab description

Add after the existing MIDI Out paragraph (around line 385). Lead with Tailscale; it is the only option that works from a phone or tablet without touching that device's trust store.

Content to cover:

- Web MIDI needs HTTPS or a loopback origin. On the server machine, `http://localhost:<port>` just works and nothing further is needed.
- **Tailscale (recommended for multi-device):** `tailscale serve 8765` gives a publicly trusted, auto-renewed cert on a `*.ts.net` hostname. No browser warnings, no CA distribution, works on mobile.
- **SSH tunnel (no certificates at all):** `ssh -L 8765:localhost:8765 user@server`, then browse `http://localhost:8765` on the client. The origin genuinely is loopback. Fastest option for a one-off test from another desktop.
- **Built-in TLS:** generate a cert with [mkcert](https://github.com/FiloSottile/mkcert), install the local CA on each client machine, then run `uv run stemforge --ssl-certfile cert.pem --ssl-keyfile key.pem`. Note plainly that mobile CA installation is painful, and that a bare self-signed cert without a trusted CA will not produce a usable secure context in Chrome or Brave — the click-through warning page is not sufficient.
- One line noting HF Spaces already serves HTTPS, so MIDI Out works there with no setup.

### 4.2 · README — line 385 correction

The sentence ending "Requires Chrome or Edge and a secure context (localhost or HTTPS)" should link forward to the new subsection rather than leaving the reader to work out what a secure context is.

---

## 5 · DO NOT

- **Do not remove, weaken, or work around the `window.isSecureContext` check** in `midi.js` (lines 674 and 744). It is correct and it is the browser's contract.
- **Do not generate certificates.** No `mkcert` invocation, no `cryptography`-based self-signed cert generation, no `openssl` subprocess. StemForge consumes cert paths; it does not produce certs.
- **Do not add a `--no-verify-ssl`, `--insecure`, or equivalent escape hatch.**
- **Do not change the bind host** from `0.0.0.0`, and do not add a `--host` flag. Out of scope.
- **Do not touch `backend/main.py`**, the `/api/midi/events` endpoint, `frontend/components/webmidi.js`, or any scheduler logic. TLS terminates in uvicorn and is invisible above it.
- **Do not add new dependencies.** Uvicorn's TLS support comes from the stdlib `ssl` module and is already installed.
- **Do not silently fall back to HTTP** when only one of the two flags is supplied.
- **Do not attempt to auto-detect the LAN IP** and print it in the banner. It invites users onto exactly the insecure origin this spec is working around.

---

## 6 · Testing

### 6.1 · Automated

Add to `tests/test_midi_events.py` or a new `tests/test_cli_args.py`:

1. `_parse_args` accepts both SSL flags and stores them.
2. Both flags absent → `args.ssl_certfile is None and args.ssl_keyfile is None`.
3. The mismatched-flag validation exits non-zero. Test the predicate directly rather than invoking `main()`, which acquires the GPU lock.

Do not attempt an end-to-end TLS handshake test — that needs a real cert and belongs in manual validation.

### 6.2 · Manual

1. **No flags:** `uv run stemforge` → banner shows `http://localhost:8765`, app works, MIDI Out works on the server machine. Confirms zero regression.
2. **One flag only:** `uv run stemforge --ssl-certfile cert.pem` → clear error, exit 1, no GPU lock file left in `~/.local/share/stemforge/`.
3. **Both flags, mkcert cert:** banner shows `https://localhost:8765`. From a *second machine* with the mkcert CA installed, browse `https://<server-ip>:8765` → MIDI tab shows "Click Request access", not the secure-context error. Grant access, confirm hardware ports enumerate and a stem plays to a hardware synth.
4. **SSH tunnel path:** with no TLS flags, tunnel from a second machine and confirm MIDI Out works at `http://localhost:8765` on the client.

Step 3 is the acceptance test. Steps 1 and 2 guard the regression surface.

---

## 7 · Effort estimate

- `run.py`: ~20 lines across four edits
- `midi.js`: one string
- Tests: ~25 lines
- README: one subsection plus a one-line correction

Half a session. The documentation is the larger share, and is the part that actually resolves the user-facing problem — the flags alone would not have prevented the 2026-08-30 misdiagnosis, but the banner fix and README section would have.

---

## 8 · Definition of Done

1. `run.py` exposes `--ssl-certfile` and `--ssl-keyfile`, both defaulting to `None`.
2. Supplying exactly one of the two prints an actionable error and exits 1, before the GPU lock is acquired.
3. `_print_banner` takes a `scheme` parameter defaulting to `"http"` and uses it in the Server line.
4. The banner prints `https://` when `--ssl-certfile` is set.
5. `uvicorn.run` receives both SSL parameters unconditionally; `None`/`None` preserves current behavior exactly.
6. `midi.js` line 675 no longer hardcodes port 8765 and no longer names loopback as the only remedy.
7. No other frontend file is modified.
8. `grep -rn "ws://\|WebSocket\|EventSource" frontend/` still returns nothing.
9. No new entries in `pyproject.toml` dependencies.
10. README documents Tailscale, SSH tunnel, and built-in TLS, in that order, under the MIDI tab section.
11. README notes that HF Spaces already provides HTTPS.
12. README states that a bare self-signed cert is insufficient in Chrome/Brave.
13. Existing test suite passes unchanged.
14. Manual validation steps 1–4 in §6.2 all pass.
