"""StemForge launcher — starts the FastAPI server and optionally the AceStep subprocess."""

import os
import sys

# ── jemalloc ──────────────────────────────────────────────────────────────────
# Re-exec with LD_PRELOAD=libjemalloc if available and not already active.
# Improves long-running stability by reducing heap fragmentation under
# concurrent multi-pipeline workloads.  Silent no-op if jemalloc is absent.
# Set STEMFORGE_NO_JEMALLOC=1 to opt out.
if sys.platform == "linux" and "STEMFORGE_NO_JEMALLOC" not in os.environ:
    _JEMALLOC_CANDIDATES = [
        "/usr/lib64/libjemalloc.so.2",
        "/usr/lib/libjemalloc.so.2",
        "/usr/local/lib/libjemalloc.so.2",
    ]
    _jl = next((p for p in _JEMALLOC_CANDIDATES if os.path.exists(p)), None)
    if _jl and "jemalloc" not in os.environ.get("LD_PRELOAD", ""):
        _env = os.environ.copy()
        _env["LD_PRELOAD"] = _jl
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import atexit
import fcntl
import signal
import subprocess
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

# Cross-process GPU lock — prevents StemForge and Wrangler from running
# simultaneously.  Uses fcntl.flock() which is kernel-enforced and
# automatically released on process exit (even on crash).
_GPU_LOCK_PATH = Path.home() / ".local" / "share" / "stemforge" / "gpu.lock"


def _acquire_gpu_lock() -> object:
    """Acquire exclusive GPU lock.  Returns the open file handle (must stay open)."""
    _GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(_GPU_LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Another app holds the lock — read who
        try:
            with open(_GPU_LOCK_PATH) as rf:
                holder = rf.read().strip() or "another GPU application"
        except OSError:
            holder = "another GPU application"
        print(f"\n  ERROR: GPU is locked by {holder}")
        print("  Only one GPU application (StemForge or Wrangler) can run at a time.")
        print("  Stop the other application first, then try again.\n")
        fh.close()
        sys.exit(1)
    fh.write(f"StemForge (PID {os.getpid()})\n")
    fh.flush()
    return fh

load_dotenv(override=False)

# AceStep passthrough vars listed here only for the banner display.
_ACESTEP_PASSTHROUGH_VARS = [
    "ACESTEP_DEVICE",
    "MAX_CUDA_VRAM",
    "ACESTEP_VAE_ON_CPU",
    "ACESTEP_LM_BACKEND",
    "ACESTEP_INIT_LLM",
    "ACESTEP_NO_INIT",
    "MODEL_LOCATION",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch StemForge + optional AceStep server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("STEMFORGE_PORT", "8765")),
        help="StemForge server port (default: 8765, env STEMFORGE_PORT)",
    )
    parser.add_argument(
        "--acestep-port",
        type=int,
        default=int(os.environ.get("ACESTEP_PORT", "8001")),
        help="AceStep API server port (default: 8001, env ACESTEP_PORT)",
    )
    parser.add_argument(
        "--no-acestep",
        action="store_true",
        default=False,
        help="Disable the AceStep subprocess (Compose tab unavailable). "
             "Equivalent to --compose-mode disabled.",
    )
    parser.add_argument(
        "--compose-mode",
        choices=["embedded", "remote", "disabled"],
        default=os.environ.get("COMPOSE_MODE", "embedded"),
        help="Compose backend mode: embedded (default, local subprocess), "
             "remote (standalone Wrangler at --compose-url), or disabled.",
    )
    parser.add_argument(
        "--compose-url",
        type=str,
        default=os.environ.get("COMPOSE_URL"),
        help="URL of a remote Wrangler instance for --compose-mode remote "
             "(e.g. http://192.168.1.50:8001).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=os.environ.get("ACESTEP_GPU"),
        help="GPU device(s) for AceStep (e.g. 0, 1, 0,1). Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("MODEL_LOCATION", "").strip() or None,
        help="Shared model cache directory (also MODEL_LOCATION env var). "
             "Default: ~/.cache/stemforge/",
    )
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic generation: low LM temperature + CUDA "
             "deterministic ops when a seed is set. Useful for A/B testing.",
    )
    # Multi-user settings
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Max concurrent users (0=unlimited, default: env MAX_USERS or 0)",
    )
    parser.add_argument(
        "--max-jobs-per-user",
        type=int,
        default=None,
        help="Max pending jobs per user (default: env MAX_JOBS_PER_USER or 3)",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=None,
        help="Session inactivity timeout in minutes (default: env or 60)",
    )
    parser.add_argument(
        "--job-ttl",
        type=int,
        default=None,
        help="Completed job TTL in minutes (default: env or 120)",
    )
    return parser.parse_args()


def _print_banner(
    port: int,
    acestep_port: int,
    compose_mode: str,
    gpu: str | None,
    model_dir: str,
    compose_url: str | None = None,
    scheme: str = "http",
) -> None:
    gpu_display = gpu if gpu else "auto"
    if compose_mode == "embedded":
        compose_display = f"embedded (port {acestep_port}, starts on first use)"
    elif compose_mode == "remote":
        compose_display = f"remote ({compose_url})"
        gpu_display = f"{gpu_display} (AceStep on remote host)"
    else:
        compose_display = "disabled"
    active_overrides = {
        k: os.environ[k] for k in _ACESTEP_PASSTHROUGH_VARS if k in os.environ
    }
    print()
    print("=" * 60)
    print("  StemForge")
    print("=" * 60)
    print(f"  Server:     {scheme}://localhost:{port}")
    print(f"  Models:     {model_dir}")
    print(f"  Compose:    {compose_display}")
    print(f"  GPU:        {gpu_display}")
    max_users = int(os.environ.get("MAX_USERS", "0"))
    if max_users > 0:
        session_timeout = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "60"))
        print(f"  Users:      max {max_users} (timeout {session_timeout}m)")
    if active_overrides and compose_mode == "embedded":
        print("-" * 60)
        print("  AceStep env overrides:")
        for k, v in active_overrides.items():
            print(f"    {k}={v}")
    print("=" * 60)
    print()


def main() -> None:
    args = _parse_args()

    # TLS flags travel in pairs — a lone flag means the user wants TLS, so
    # refuse rather than silently start over HTTP. Checked before the GPU
    # lock so a config error can't leave a lock file behind.
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        print(
            "\n  ERROR: --ssl-certfile and --ssl-keyfile must be given together.\n"
            "  TLS is required for Web MIDI Out from a non-localhost browser.\n"
            "  See the MIDI Out section of README.md for certificate options.\n"
        )
        sys.exit(1)

    # --- GPU exclusion lock (must be first — before any heavy imports) ---
    _gpu_lock_fh = _acquire_gpu_lock()  # noqa: F841  (must stay open)

    # --- Model cache directory (must be set before any model imports) ---
    if args.model_dir:
        os.environ["MODEL_LOCATION"] = args.model_dir
    # Also redirect torch.hub (used internally by Demucs) into our cache tree.
    from utils.cache import get_model_cache_base
    model_base = get_model_cache_base()
    os.environ.setdefault("TORCH_HOME", str(model_base / "torch_hub"))

    if args.deterministic:
        os.environ["STEMFORGE_DETERMINISTIC"] = "1"

    # Forward multi-user CLI args as env vars (before app import)
    if args.max_users is not None:
        os.environ["MAX_USERS"] = str(args.max_users)
    if args.max_jobs_per_user is not None:
        os.environ["MAX_JOBS_PER_USER"] = str(args.max_jobs_per_user)
    if args.session_timeout is not None:
        os.environ["SESSION_TIMEOUT_MINUTES"] = str(args.session_timeout)
    if args.job_ttl is not None:
        os.environ["JOB_TTL_MINUTES"] = str(args.job_ttl)

    # --no-acestep takes precedence over --compose-mode for backward compat.
    compose_mode_str = "disabled" if args.no_acestep else args.compose_mode

    from backend.compose_backend import configure_compose_backend
    from backend.compose_backend.protocol import BackendMode

    configure_compose_backend(
        mode=BackendMode(compose_mode_str),
        port=args.acestep_port,
        gpu=args.gpu,
        remote_url=args.compose_url,
    )

    # Disabled mode: also set acestep_state directly so the existing compose
    # router (pre-Step-7) reports "disabled" correctly via its _require_acestep().
    if compose_mode_str == "disabled":
        from backend.services import acestep_state
        acestep_state.set_status("disabled", port=args.acestep_port)

    _print_banner(
        args.port, args.acestep_port, compose_mode_str,
        args.gpu, str(model_base), args.compose_url,
        scheme="https" if args.ssl_certfile else "http",
    )

    # Graceful shutdown: terminate AceStep subprocess (embedded mode only).
    # acestep_state is already configured by EmbeddedComposeBackend.__init__.
    def _kill_acestep() -> None:
        from backend.services import acestep_state
        proc = acestep_state.get_process()
        if proc and proc.poll() is None:
            print("\n[stemforge] Stopping AceStep...")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()

    # atexit runs when Python exits — catches cases where uvicorn
    # swallows SIGINT and shuts down on its own without calling _shutdown
    atexit.register(_kill_acestep)

    def _shutdown(signum: int, frame: object) -> None:
        _kill_acestep()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start StemForge FastAPI server (blocks)
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
