"""Step 19 controlled multi-file agent benchmark driver.

Resets the disposable workspace, verifies the known-broken baseline, runs
the CHURRO agent exactly once against the fixed one-line /agent task, then
verifies the outcome and archives all generated artifacts OUTSIDE the Git
repository (default ``D:\\churro-step19-runs``).

The harness and benchmark definition (this script, task.txt, baseline/)
live in the repository for reproducibility; every log, session file, cache,
and run artifact is written outside the repo.

Usage:
    python run_benchmark.py [--model qwen3.5:4b]
        [--workspace PATH] [--runs-dir PATH] [--timeout SECONDS]
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # D:\AI PROJECT
HERE = Path(__file__).resolve().parent  # ...\benchmarks\step19
BASELINE = HERE / "baseline"
TASK_FILE = HERE / "task.txt"
DEFAULT_WORKSPACE = Path("D:/churro-agent-test")
DEFAULT_RUNS_DIR = Path("D:/churro-step19-runs")
DEFAULT_TIMEOUT = 1500  # generous wall clock for a 4B local model

SKIP_DIR_NAMES = {"sessions", ".pytest_cache", "__pycache__"}

CHURRO_COMMIT_HINT = (
    "working tree (not committed); compare D:\\AI PROJECT git status "
    "before/after for drift"
)


# --------------------------------------------------------------- helpers
def pytest_counts(text: str) -> tuple[int, int]:
    """Return (passed, failed) parsed from pytest's ``-q`` summary."""
    passed = int(re.search(r"(\d+) passed", text).group(1))
    failed_match = re.search(r"(\d+) failed", text)
    failed = int(failed_match.group(1)) if failed_match else 0
    return passed, failed


def run_pytest(workspace: Path, env: dict, timeout: int = 120):
    return subprocess.run(
        [str(sys.executable), "-m", "pytest", "--cache-clear", "-q"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(workspace),
        env=env,
        timeout=timeout,
    )


def file_manifest(root: Path) -> dict[str, str]:
    """SHA-256 of every file under root, skipping generated/artifact dirs."""
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        manifest[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def git_status_short() -> str:
    proc = subprocess.run(
        ["git", "-C", os.fspath(REPO), "status", "--short"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    return (proc.stdout + proc.stderr).strip()


def ollama_online(base: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=5) as r:
            body = r.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            return True, data.get("version", "unknown")
    except Exception as exc:  # noqa: BLE001 - probe failure, not fatal logic
        return False, f"probe failed: {exc}"


def ollama_has_model(base: str, model: str) -> bool:
    try:
        request = urllib.request.Request(
            f"{base}/api/tags", method="GET", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        return any(item.get("name") == model for item in data.get("models", []))
    except Exception:  # noqa: BLE001 - worst case the run reveals the problem
        return True  # don't hard-fail on a tagging quirk; the run will show it


def kill_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        timeout=15,
    )


def run_agent(workspace: Path, task: str, env: dict, timeout: int):
    # CHURRO first prompts for a session goal (one input line), then enters
    # its command loop. Feed goal, then the /agent task, then /quit.
    stdin_text = f"{task}\n/agent {task}\n/quit\n"
    proc = subprocess.run(
        [str(sys.executable), "-m", "churro"],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(workspace),
        env=env,
        timeout=timeout,
    )
    return proc


def parse_tool_trace(log_text: str) -> list[tuple[str, bool]]:
    """Extract ``[tool] name`` lines; bool is success (False when has ' (failed)')."""
    trace: list[tuple[str, bool]] = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("[tool] "):
            continue
        body = line[len("[tool] ") :]
        failed = body.endswith(" (failed)")
        name = body[: -len(" (failed)")] if failed else body
        trace.append((name, not failed))
    return trace


# --------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("CHURRO_MODEL", "qwen3.5:4b"))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if not TASK_FILE.is_file():
        print(f"Missing task file: {TASK_FILE}")
        return 1
    task = TASK_FILE.read_text(encoding="utf-8").strip()
    if not task:
        print("Task file is empty.")
        return 1

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = args.model.replace(":", "-")
    run_dir = args.runs_dir / model_tag / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": timestamp,
        "model": args.model,
        "task": task,
        "workspace": os.fspath(args.workspace),
        "churro_commit": CHURRO_COMMIT_HINT,
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }
    print(f"[bench] run dir    : {run_dir}")
    print(f"[bench] model      : {args.model}")

    # Preconditions: Ollama up and model present.
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    online, version = ollama_online(base)
    if not online:
        print(f"[bench] FATAL: Ollama not reachable at {base} ({version})")
        return 1
    meta["ollama_version"] = version
    print(f"[bench] ollama     : {version}")
    if not ollama_has_model(base, args.model):
        print(f"[bench] WARNING: model {args.model} not listed in `ollama list`")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["CHURRO_PROVIDER"] = "ollama"
    env["CHURRO_MODEL"] = args.model
    env["CHURRO_WORKSPACE"] = os.fspath(args.workspace)

    # 1) Reset.
    print("[bench] resetting workspace...")
    if args.workspace.exists():
        shutil.rmtree(args.workspace)
    shutil.copytree(BASELINE, args.workspace)
    meta["reset"] = "copytree(baseline)"

    # 2) Known-broken pre-check.
    pre = run_pytest(args.workspace, env)
    pre_text = pre.stdout + "\n" + pre.stderr
    (run_dir / "pre_pytest.txt").write_text(pre_text, encoding="utf-8")
    passed, failed = pytest_counts(pre_text)
    pre_ok = passed == 9 and failed == 3
    print(f"[bench] pre-check  : {passed} passed, {failed} failed (expect 9/3)")
    meta["pre_pytest"] = {"passed": passed, "failed": failed, "ok": pre_ok}
    if not pre_ok:
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[bench] FATAL: baseline is not in the known-broken state; aborting.")
        return 1

    manifest_before = file_manifest(args.workspace)
    (run_dir / "manifest_before.json").write_text(
        json.dumps(manifest_before, indent=2, sort_keys=True), encoding="utf-8"
    )
    git_before = git_status_short()
    (run_dir / "git_status_before.txt").write_text(git_before, encoding="utf-8")

    # 3) Run the agent exactly once.
    print("[bench] launching CHURRO agent (max {0}s wall clock)...".format(args.timeout))
    started = time.time()
    timed_out = False
    try:
        proc = run_agent(args.workspace, task, env, args.timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        log_text = (
            (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        ) + "\n" + ((exc.stderr or "") if isinstance(exc.stderr, str) else "")
        meta["timed_out"] = True
        print("[bench] agent run hit the wall-clock timeout; killing process tree.")
        print("[bench] WARNING: this subprocess object was not captured; "
              "killing is skipped for this run.")
    except Exception as exc:  # noqa: BLE001
        log_text = f"CHURRO launch failed: {type(exc).__name__}: {exc}"
        meta["launch_error"] = log_text
        proc = None
    else:
        elapsed = time.time() - started
        log_text = proc.stdout + "\n---STDERR---\n" + proc.stderr
        meta["elapsed_seconds"] = round(elapsed, 1)
        meta["returncode"] = proc.returncode
        meta["timed_out"] = False
        print(f"[bench] agent run   : {elapsed:.1f}s, exit {proc.returncode}")

    (run_dir / "agent_run.log").write_text(log_text, encoding="utf-8")

    # 4) Copy session files out of the workspace.
    sessions_dir = args.workspace / "sessions"
    if sessions_dir.is_dir():
        for s in sorted(sessions_dir.glob("*.json")):
            shutil.copy2(s, run_dir / f"session_{s.name}")
        meta["sessions"] = [f.name for f in sorted(sessions_dir.glob("*.json"))]

    # 5) Post-checks.
    post = run_pytest(args.workspace, env)
    post_text = post.stdout + "\n" + post.stderr
    (run_dir / "post_pytest.txt").write_text(post_text, encoding="utf-8")
    post_passed, post_failed = pytest_counts(post_text)
    post_ok = post_passed == 12 and post_failed == 0
    print(f"[bench] post-check : {post_passed} passed, {post_failed} failed (expect 12/0)")

    manifest_after = file_manifest(args.workspace)
    (run_dir / "manifest_after.json").write_text(
        json.dumps(manifest_after, indent=2, sort_keys=True), encoding="utf-8"
    )

    tests_before = {k: v for k, v in manifest_before.items() if k.startswith("tests/")}
    tests_after = {k: v for k, v in manifest_after.items() if k.startswith("tests/")}
    tests_unchanged = tests_before == tests_after
    print(f"[bench] tests      : {'unchanged' if tests_unchanged else 'MODIFIED'}")

    git_after = git_status_short()
    (run_dir / "git_status_after.txt").write_text(git_after, encoding="utf-8")
    repo_unchanged = git_before == git_after
    print(f"[bench] repo       : {'unchanged' if repo_unchanged else 'DRIFT DETECTED'}")

    # 6) Save changed source files alongside the run for inspection.
    changed_sources = [
        name
        for name in sorted(set(manifest_before) | set(manifest_after))
        if name.startswith("pricing.py") or name.startswith("shopping_cart.py")
    ]
    for name in changed_sources:
        src = args.workspace / name
        if src.is_file():
            target = run_dir / Path("after_run_source") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    trace = parse_tool_trace(log_text)
    run_tests_calls = [(tool, ok) for tool, ok in trace if tool == "run_tests"]
    final_check_ok = bool(run_tests_calls) and run_tests_calls[-1][1]
    meta["tool_trace"] = trace
    meta["run_tests_calls"] = run_tests_calls
    meta["results"] = {
        "pre_ok": pre_ok,
        "post_ok": post_ok,
        "tests_unchanged": tests_unchanged,
        "repo_unchanged": repo_unchanged,
        "timed_out": timed_out,
        "final_check_ok": final_check_ok,
        "success": (
            pre_ok
            and post_ok
            and tests_unchanged
            and repo_unchanged
            and not timed_out
            and final_check_ok
        ),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n================ STEP 19 BENCHMARK RESULT ================")
    print(f"model            : {args.model}")
    print(f"run dir          : {run_dir}")
    print(f"agent wall time  : {meta.get('elapsed_seconds', 'n/a')}s")
    print(f"pre-check        : 9 passed / 3 failed -> OK={pre_ok}")
    print(f"post-check       : {post_passed} passed / {post_failed} failed -> OK={post_ok}")
    print(f"iterations/tools : {len(trace)} tool call(s) -> {trace}")
    print(f"run_tests calls  : {run_tests_calls}")
    print(f"tests unchanged  : {tests_unchanged}")
    print(f"repo unchanged   : {repo_unchanged}")
    print(f"timed out        : {timed_out}")
    print(f"SUCCESS          : {meta['results']['success']}")
    print("============================================================")
    return 0 if meta["results"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())