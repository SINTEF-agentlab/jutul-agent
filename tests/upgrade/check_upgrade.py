"""Upgrade a real jutul-agent install to Makker, in a folder of its own, and check it.

Temporary, like ``makker.legacy`` and ``bridge/``: it goes when they do.

    uv run --no-project tests/upgrade/check_upgrade.py --from 0.2.1

It installs the named jutul-agent release from PyPI as a uv tool, gives it a
data folder and a workspace the way that release left them, builds Makker and
the bridge from this checkout, and then does what a user's machine does:
``uv tool upgrade jutul-agent``, followed by starting ``jutul-agent`` again.

With ``--published`` it skips the local build and takes Makker and the bridge
from PyPI instead, which is the check to run once a release is out. With
``--manual`` it stops after the set-up and says how to do the update by hand,
for trying the Update button of the old web UI.

The local build is given the version of a final release. A release candidate
would not do: ``uv tool upgrade`` only takes one when pre-releases are allowed
for every package, and that pulls in development builds of dependencies.

Nothing outside ``--root`` is touched: uv's tool and executable folders and the
data folder are all pointed inside it. No Julia is needed, so the part of the
move that rebuilds a workspace's Julia environment is not exercised here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WINDOWS = os.name == "nt"
NEXT = "0.3.0"  # the version the local build stands in for

OLD_PROJECT = """\
[deps]
JutulAgent = "79b12aa6-73c9-4fa8-9c54-e97745f42687"
JutulAgentJutulDarcy = "69df87d8-8b4b-4157-81d2-8b93ff139141"
JutulDarcy = "82210473-ab04-4dce-b31b-11573c4f8e0a"

[sources]
JutulAgent = {path = "JutulAgent"}
JutulAgentJutulDarcy = {path = "JutulAgentJutulDarcy"}
"""


def step(text: str) -> None:
    print(f"\n== {text}", flush=True)


def run(argv: list[str], env: dict[str, str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(str(a) for a in argv), flush=True)
    done = subprocess.run(argv, env=env, text=True, capture_output=True, **kwargs)
    for stream in (done.stdout, done.stderr):
        if stream.strip():
            print("    " + stream.strip().replace("\n", "\n    "), flush=True)
    return done


def check(condition: bool, what: str) -> None:
    print(("  ok    " if condition else "  FAILED ") + what, flush=True)
    if not condition:
        raise SystemExit(f"check failed: {what}")


def command(bin_dir: Path, name: str) -> Path:
    return bin_dir / (name + ".exe" if WINDOWS else name)


def lay_out_old_data(data: Path, workspace: Path) -> None:
    step("give it the data folder and workspace that release leaves behind")
    old_home = data / "jutul-agent"
    memory = old_home / "workspaces" / "0123456789ab" / "memory"
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text("- a note from before\n", encoding="utf-8")
    (old_home / ".env").write_text("OPENAI_API_KEY=sk-kept\n", encoding="utf-8")
    env_dir = workspace / ".jutul-agent" / "julia-env"
    for package in ("JutulAgent", "JutulAgentJutulDarcy"):
        (env_dir / package / "src").mkdir(parents=True)
        (env_dir / package / "Project.toml").write_text(f'name = "{package}"\n', encoding="utf-8")
    (env_dir / "Project.toml").write_text(OLD_PROJECT, encoding="utf-8")
    (workspace / ".jutul-agent" / "config.toml").write_text(
        '[workspace]\nsimulator = "jutuldarcy"\n', encoding="utf-8"
    )
    old_output = workspace / "jutul-agent-output" / "sessions" / "2026-10-01-1200-ab12"
    old_output.mkdir(parents=True)
    (old_output / "report.html").write_text("<p>an old report</p>", encoding="utf-8")


def hand_over(root: Path, env: dict[str, str], old: str) -> int:
    """Leave an installed old version ready for its own update to be tried by hand."""

    cache = root / "data" / "jutul-agent" / "update-check.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"checked_at": time.time(), "latest": NEXT}), encoding="utf-8")
    names = ("UV_TOOL_DIR", "UV_TOOL_BIN_DIR", "XDG_DATA_HOME", "UV_FIND_LINKS")
    settings = {name: env[name] for name in names if name in env}
    if WINDOWS:
        lines = [f'$env:{name} = "{value}"' for name, value in settings.items()]
        lines.append(f'$env:PATH = "{root / "bin"};$env:PATH"')
    else:
        lines = [f'export {name}="{value}"' for name, value in settings.items()]
        lines.append(f'export PATH="{root / "bin"}:$PATH"')
    print(
        f"""
jutul-agent {old} is installed in {root}, and believes {NEXT} is out.
In a new terminal, point it at this folder:

  {chr(10).join("  " + line for line in lines).strip()}

Then, in a scratch folder:

  jutul-agent init --sim jutuldarcy
  jutul-agent

Enter an API key when asked, have a chat that makes a plot, and click Update in
the sidebar. Delete {root} when you are done.""",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--from", dest="old", default="0.2.1", help="jutul-agent release to start from"
    )
    parser.add_argument("--root", type=Path, help="folder to work in (default: a temporary one)")
    parser.add_argument(
        "--published", action="store_true", help="upgrade to what is on PyPI, not to this checkout"
    )
    parser.add_argument(
        "--manual", action="store_true", help="set up, then leave the update to be done by hand"
    )
    args = parser.parse_args()

    root = (args.root or Path(tempfile.mkdtemp(prefix="makker-upgrade-"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bin_dir, data, links, workspace = root / "bin", root / "data", root / "links", root / "ws"
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(root / "tools"),
        "UV_TOOL_BIN_DIR": str(bin_dir),
        "XDG_DATA_HOME": str(data),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "MAKKER_NO_UPDATE_CHECK": "1",
        "JUTUL_AGENT_NO_UPDATE_CHECK": "1",
        "JUTUL_AGENT_MODEL": "openai:from-the-old-setting",
    }
    for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(name, None)

    step(f"install jutul-agent {args.old} from PyPI")
    done = run(["uv", "tool", "install", f"jutul-agent=={args.old}"], env)
    check(
        done.returncode == 0 and command(bin_dir, "jutul-agent").is_file(), "the old command exists"
    )

    if not args.manual:
        lay_out_old_data(data, workspace)
    old_home = data / "jutul-agent"
    old_output = workspace / "jutul-agent-output" / "sessions" / "2026-10-01-1200-ab12"
    step("make it an ordinary install that takes the next release")
    # Installed by exact version above, which uv would keep it at. People install
    # without one.
    receipt = root / "tools" / "jutul-agent" / "uv-receipt.toml"
    pinned = f', specifier = "=={args.old}"'
    text = receipt.read_text(encoding="utf-8")
    check(pinned in text, "the receipt records the exact version")
    receipt.write_text(text.replace(pinned, ""), encoding="utf-8")

    upgrade_env = env
    if not args.published:
        step(f"build Makker and the bridge from this checkout, as version {NEXT}")
        links.mkdir(exist_ok=True)
        build_env = {**env, "SETUPTOOLS_SCM_PRETEND_VERSION": NEXT}
        built = run(["uv", "build", "--wheel", "--out-dir", str(links), str(REPO)], build_env)
        check(built.returncode == 0, "Makker builds")
        bridge = root / "bridge-src"
        shutil.copytree(REPO / "bridge" / "jutul-agent", bridge, dirs_exist_ok=True)
        project = (bridge / "pyproject.toml").read_text(encoding="utf-8")
        versioned = re.sub(r'(?m)^version = "[^"]+"$', f'version = "{NEXT}"', project)
        versioned = re.sub(r'"makker>=[^"]+"', f'"makker>={NEXT}"', versioned)
        (bridge / "pyproject.toml").write_text(versioned, encoding="utf-8")
        built = run(["uv", "build", "--wheel", "--out-dir", str(links), str(bridge)], env)
        check(built.returncode == 0, "the bridge builds")
        upgrade_env = {**env, "UV_FIND_LINKS": str(links)}

    if args.manual:
        return hand_over(root, upgrade_env, args.old)

    step("update the way every released version does")
    done = run(["uv", "tool", "upgrade", "jutul-agent"], upgrade_env)
    check(done.returncode == 0, "uv tool upgrade jutul-agent succeeds")
    check(not command(bin_dir, "makker").exists(), "the update alone puts no `makker` on the PATH")

    step("start jutul-agent again, as its updater and its users do")
    done = run(
        [str(command(bin_dir, "jutul-agent")), "sessions", "--workspace", str(workspace)],
        upgrade_env,
    )
    check(done.returncode == 0, "it runs")
    check("jutul-agent is now Makker" in done.stderr, "it says what happened")
    check(command(bin_dir, "makker").is_file(), "Makker has its own install and command")
    listed = run(["uv", "tool", "list"], env).stdout
    check("makker" in listed, "uv lists the makker tool")

    step("the data came along")
    new_home = data / "makker"
    check(
        (new_home / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-kept\n",
        "the API key is in the new data folder",
    )
    check(
        (old_home / "workspaces" / "0123456789ab" / "memory" / "MEMORY.md").is_file(),
        "the old path still reaches the memory notes",
    )
    check(
        old_home.is_symlink() or os.path.isjunction(old_home),
        "the old data folder is a link to the new one",
    )

    step("the workspace came along")
    check((workspace / ".makker" / "config.toml").is_file(), ".makker/ holds the workspace config")
    check(not (workspace / ".jutul-agent").exists(), ".jutul-agent/ is gone")
    check(
        (workspace / ".makker" / ".gitignore").read_text(encoding="utf-8") == "*\n",
        ".makker/ keeps itself out of git",
    )
    check((old_output / "report.html").is_file(), "old output stays where it was written")

    step("start Makker under its own name")
    done = run([str(command(bin_dir, "makker")), "sessions", "--workspace", str(workspace)], env)
    check(done.returncode == 0, "it runs")
    check("JUTUL_AGENT_MODEL" in done.stderr, "the setting under the old name is pointed out")
    listed = run(["uv", "tool", "list"], env).stdout
    check("jutul-agent" not in listed, "the old install has been removed")
    check(not command(bin_dir, "jutul-agent").exists(), "and so has the old command")

    step("Makker can update itself from here")
    done = run(
        [str(command(bin_dir, "makker")), "upgrade", "--check", "--workspace", str(workspace)],
        {**env, "MAKKER_NO_UPDATE_CHECK": ""},
    )
    check(done.returncode == 0, "`makker upgrade --check` runs")

    print("\nAll checks passed.", flush=True)
    if args.root is None:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
