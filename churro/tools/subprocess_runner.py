import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CapturedResult:
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None


def run_captured(
    command: list[str] | str,
    cwd: Path,
    timeout_seconds: int,
    *,
    shell: bool = False,
    env_extra: dict[str, str] | None = None,
) -> CapturedResult:
    """Run a command and capture its output without ever raising.

    ``command`` is either an argument list (``shell=False``) or a shell
    string (``shell=True``), executed with ``cwd`` as the working directory.
    Non-zero exits, timeouts, and launch failures are all surfaced inside
    the returned ``CapturedResult`` so callers control how each becomes a
    ``ToolResult``. On timeout the whole process tree is terminated and any
    output already produced is retained.
    """
    kwargs = {
        "cwd": str(cwd),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        if shell:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                proc = subprocess.Popen(command, shell=True, **kwargs)
        else:
            proc = subprocess.Popen(command, **kwargs)
    except FileNotFoundError:
        return CapturedResult(error="Command or working directory not found")
    except OSError as exc:
        return CapturedResult(
            error=f"Failed to launch command: {type(exc).__name__}: {exc}"
        )
    except Exception as exc:
        return CapturedResult(
            error=f"Unexpected error launching command: {type(exc).__name__}: {exc}"
        )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_tree(proc)
        return CapturedResult(
            stdout=exc.output or "",
            stderr=getattr(exc, "stderr", None) or "",
            timed_out=True,
        )
    except Exception as exc:
        terminate_tree(proc)
        return CapturedResult(
            error=f"Unexpected subprocess error: {type(exc).__name__}: {exc}"
        )

    return CapturedResult(returncode=proc.returncode, stdout=stdout, stderr=stderr)


def terminate_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and, on Windows, its whole process tree."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
            )
            return
        except Exception:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass