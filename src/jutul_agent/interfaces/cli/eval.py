"""``jutul-agent eval`` subcommand: run bench suites through Inspect AI.

A thin wrapper over ``inspect eval`` that loads provider keys the way the
app does (the global jutul-agent ``.env`` plus the working directory's)
*before* Inspect resolves the target model, points logs at a stable
location, and defaults to one sample at a time; the agent under test uses
process-global workspace state, so samples must not run concurrently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _tasks_dir() -> Path:
    from jutul_agent.eval import tasks

    return Path(tasks.__file__).resolve().parent


# Inspect names a few providers differently than langchain's init_chat_model.
_INSPECT_PROVIDERS = {"google_genai": "google"}


def _default_inspect_model() -> str:
    """The agent's default model, in Inspect's ``provider/model`` form."""
    from jutul_agent.models import DEFAULT_MODEL

    provider, _, model = DEFAULT_MODEL.partition(":")
    return f"{_INSPECT_PROVIDERS.get(provider, provider)}/{model}"


def _available_suites() -> dict[str, str]:
    """Suite name -> first docstring line, for ``--list``."""
    suites: dict[str, str] = {}
    for path in sorted(_tasks_dir().glob("*.py")):
        if path.name.startswith("_"):
            continue
        title = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().strip('"')
            if stripped:
                title = stripped
                break
        suites[path.stem] = title
    return suites


def build_parser(prog: str = "jutul-agent eval") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run jutul-bench suites through Inspect AI.",
        epilog=(
            "Examples: `jutul-agent eval canary`; "
            "`jutul-agent eval canary guardrails --model <provider/model>,<provider/model>`. "
            "View results with `uv run inspect view --log-dir <dir>`."
        ),
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Suite names (see --list) or paths to Inspect task files.",
    )
    parser.add_argument(
        "--model",
        help="Inspect model id(s) as provider/model, comma-separated for a "
        "matrix. Defaults to the agent's default model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Repetitions per sample, for variance (default 1).",
    )
    parser.add_argument(
        "--epochs-reducer",
        help="How to combine scores across epochs, comma-separated Inspect "
        "reducer names: mean, median, max, pass_at_2, at_least_2, ...",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Concurrent samples (default 1; the agent's workspace state is "
        "process-global, so higher values only interleave sessions).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N samples of each task.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Where Inspect writes eval logs (default: jutul-agent home /eval-logs).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_suites",
        help="List available suites and exit.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        import jutul_agent.eval  # noqa: F401
    except ImportError:
        print(
            "error: the eval extra is not installed. Run: uv sync --extra eval",
            file=sys.stderr,
        )
        return 2

    if args.list_suites:
        for name, title in _available_suites().items():
            print(f"{name:12} {title}")
        return 0

    if not args.tasks:
        print("error: name at least one suite (see --list).", file=sys.stderr)
        return 2
    model = args.model or _default_inspect_model()

    # Keys must be in the environment before Inspect resolves the model.
    from jutul_agent.credentials import load_user_credentials

    load_user_credentials()

    # Built-in suites are imported and instantiated (Inspect's file loader
    # only takes paths relative to the working directory); a path argument
    # is handed to Inspect as the user typed it.
    import importlib

    resolved: list[Any] = []
    for name in args.tasks:
        if Path(name).exists():
            resolved.append(name)
            continue
        if (_tasks_dir() / f"{name}.py").exists():
            module = importlib.import_module(f"jutul_agent.eval.tasks.{name}")
            # A suite module exposes one task named like the module, or
            # several through a TASKS list of task factories.
            factories = getattr(module, "TASKS", None) or [getattr(module, name)]
            resolved.extend(factory() for factory in factories)
            continue
        print(f"error: unknown suite or path: {name} (see --list).", file=sys.stderr)
        return 2

    from jutul_agent.paths import state_home

    log_dir = args.log_dir or (state_home() / "eval-logs")

    from inspect_ai import Epochs
    from inspect_ai import eval as inspect_eval

    epochs: int | Epochs = args.epochs
    if args.epochs_reducer:
        epochs = Epochs(args.epochs, args.epochs_reducer.split(","))

    logs = inspect_eval(
        tasks=resolved,
        model=model.split(","),
        epochs=epochs,
        max_samples=args.max_samples,
        limit=args.limit,
        log_dir=str(log_dir),
    )
    print(f"\nLogs: {log_dir} (view with `uv run inspect view --log-dir {log_dir}`)")

    # Write each sample's pass/fail back onto its session so the run can be reviewed
    # against ground truth (`jutul-agent review`). Best-effort: never fail over it.
    try:
        from jutul_agent.review.eval_link import link_eval_results

        linked = link_eval_results(logs)
        if linked:
            print(f"Linked eval verdicts onto {linked} session(s) for `jutul-agent review`.")
    except Exception:
        pass

    return 0 if all(log.status == "success" for log in logs) else 1
