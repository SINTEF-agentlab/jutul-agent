"""Custom Julia system images: where one lives, and whether it still matches the env.

A system image removes the one cost precompilation cannot touch. Loading several
hundred pkgimages is most of a cold start, and it is not compilation: it is
opening, validating and mapping that many cache files. A system image is all of
them in one already-linked file the process starts from.

The price is that a system image is not checked against anything. ``PackageCompiler``
builds it by ``Base.require``-ing each package and binding it into ``Main``, so at
run time ``using JutulDarcy`` finds it already loaded. A pkgimage revalidates
against its source on every load; a package inside a system image never does. An
edit to a baked package, or a different version resolved into the manifest, does
not fail. It runs the old code.

So this module exists to answer one question before a session starts: does the
image on disk still describe this environment? It answers it from files alone
(the manifest, and the sources of every path-tracked package), which is what keeps
the check cheap enough to run on every launch.

Two kinds of mismatch, and only one is dangerous:

- **divergent.** The image holds a package whose version or source has changed.
  The session would run code that is not what is on disk. The launch is refused.
- **incomplete.** The environment has a package the image does not. Installed
  after the build, by the user or by the agent. It loads the ordinary way and
  nothing is wrong, so this is a note, never a refusal.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from jutul_agent.workspace import package_source_digest, workspace_dir

# The image and its stamp live under the workspace's own state directory rather
# than inside the Julia env, because the env may be a user-owned root project
# that we never write to. One image per workspace, at a path nothing has to name.
SYSIMAGE_DIRNAME = "sysimage"
IMAGE_STEM = "jutul-agent-sys"
STAMP_FILENAME = "sysimage.json"

# Version of the build procedure itself. Bump it whenever how an image is built
# changes, so an upgrade retires images made the old way instead of trusting them.
RECIPE_VERSION = 1

SYSIMAGE_ENV_VAR = "JUTUL_AGENT_SYSIMAGE"
XVFB_OPT_OUT = "JUTUL_AGENT_NO_XVFB"

# Packages that bind a window system as they initialise, and so decide whether the
# process can start at all once they are baked into an image.
OPENGL_PACKAGES = frozenset({"GLFW", "GLMakie"})

# Statuses, in the order they are checked.
OFF = "off"
MISSING = "missing"
UNUSABLE = "unusable"
DIVERGENT = "divergent"
INCOMPLETE = "incomplete"
CURRENT = "current"


def on_windows() -> bool:
    """Whether images here are Windows DLLs. A function so tests can simulate
    Windows by patching it: patching ``os.name`` itself turns every ``Path``
    constructed afterwards into an uninstantiable ``WindowsPath`` on POSIX."""
    return os.name == "nt"


def image_suffix() -> str:
    """The shared-library extension Julia expects for a system image on this OS."""
    if on_windows():
        return ".dll"
    return ".dylib" if sys.platform == "darwin" else ".so"


def sysimage_dir(workspace: Path) -> Path:
    return workspace_dir(workspace) / SYSIMAGE_DIRNAME


def sysimage_path(workspace: Path) -> Path:
    return sysimage_dir(workspace) / (IMAGE_STEM + image_suffix())


def stamp_path(workspace: Path) -> Path:
    return sysimage_dir(workspace) / STAMP_FILENAME


# ---------------------------------------------------------------------------
# What the environment currently is.


@dataclass(frozen=True)
class EnvState:
    """The part of a Julia environment a system image can go stale against.

    ``versions`` are the registry packages and the version the manifest resolved.
    ``path_packages`` are the ones tracked by a path (the shared ``JutulAgent``,
    each ``JutulAgent<Sim>``, every capability package, and any ``--source-path``
    checkout), mapped to a hash of their sources. Those need hashing rather than
    versioning because their whole point is that they change without a release.

    ``preferences`` digests ``LocalPreferences.toml``. A preference a package
    reads while precompiling is compiled into the image the same way its code is,
    so changing one afterwards leaves the image holding the old value while the
    manifest still matches. ``None`` when the environment has no such file.
    """

    versions: dict[str, str] = field(default_factory=dict)
    path_packages: dict[str, str] = field(default_factory=dict)
    preferences: str | None = None


def read_env_state(julia_project: Path) -> EnvState:
    """Read the environment's resolved packages straight out of ``Manifest.toml``.

    Deliberately no Julia subprocess: this runs on every launch, and everything
    needed is in the manifest. Path entries are resolved relative to the manifest,
    which is how Julia reads them, so an env-local ``path = "JutulAgent"`` and an
    absolute capability path both land on the right directory.

    An unreadable or unresolved manifest yields a state with no packages in it,
    which reads as "nothing to compare" and leaves the image looking complete.
    That is the right failure direction: a broken env has louder problems than a
    stale image, and ``prepare_workspace_env`` has already run by the time this is
    called. Preferences are read either way, being a separate file that does not
    stop making sense when the manifest does.
    """

    preferences = _preferences_digest(julia_project)
    manifest = julia_project / "Manifest.toml"
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return EnvState(preferences=preferences)

    deps = data.get("deps")
    if not isinstance(deps, dict):  # manifest format 1.0, or nothing resolved
        return EnvState(preferences=preferences)

    versions: dict[str, str] = {}
    path_packages: dict[str, str] = {}
    for name, entries in deps.items():
        if not isinstance(entries, list) or not entries:
            continue
        entry = entries[0]
        if not isinstance(entry, dict):
            continue
        source = entry.get("path")
        if isinstance(source, str) and source:
            root = Path(source)
            if not root.is_absolute():
                root = julia_project / root
            path_packages[name] = package_source_digest(root)
        elif isinstance(entry.get("version"), str):
            versions[name] = entry["version"]
    return EnvState(versions=versions, path_packages=path_packages, preferences=preferences)


def _preferences_digest(julia_project: Path) -> str | None:
    """Hash ``LocalPreferences.toml`` by what it says, not by how it is written.

    Parsed and re-serialised in a canonical order first, so reformatting or
    reordering the file is not mistaken for changing a setting. Capability
    composition rewrites this file on every launch, and it must land on the same
    digest each time or every launch would look like a change.
    """

    path = julia_project / "LocalPreferences.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def julia_version() -> str | None:
    """The version of the ``julia`` on PATH, or ``None`` if it cannot be asked.

    ``julia --version`` short-circuits before the runtime comes up (about 10ms
    here), so this is cheap enough for the launch path, unlike anything that has
    to evaluate code.
    """

    if shutil.which("julia") is None:
        return None
    try:
        result = subprocess.run(
            ["julia", "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    # "julia version 1.12.4"
    return result.stdout.strip().rsplit(" ", 1)[-1] or None


def platform_tag() -> str:
    """A coarse platform identity, from Python alone.

    A system image is native code: it is tied to the OS and the architecture that
    built it, and running one from elsewhere is a crash rather than a slowdown.
    This is the cheap stand-in for Julia's ``Sys.MACHINE``, which would cost a full
    runtime start to ask for.
    """

    return f"{sys.platform}-{platform.machine()}"


# ---------------------------------------------------------------------------
# The stamp: what the image on disk was built from.


def write_stamp(
    workspace: Path,
    julia_project: Path,
    *,
    cpu_target: str,
    build_seconds: float,
    julia: str | None = None,
) -> None:
    """Record what the freshly built image describes. Written last, on purpose.

    An image is only ever trusted through its stamp, so stamping is what promotes
    a built file to a usable one. The builder writes this after the image has been
    moved into place *and* verified, so an interrupted or broken build leaves
    something that reads as missing rather than as current.
    """

    state = read_env_state(julia_project)
    payload = {
        "recipe": RECIPE_VERSION,
        "julia": julia or julia_version(),
        "platform": platform_tag(),
        "cpu_target": cpu_target,
        # Metadata, not a check. jutul-agent's version moves with every commit,
        # while what actually has to agree with the image is the Julia it ships:
        # the shared JutulAgent package, which is copied into the env and so is
        # already covered as a path package.
        "jutul_agent_version": _installed_version(),
        "versions": dict(sorted(state.versions.items())),
        "path_packages": dict(sorted(state.path_packages.items())),
        "preferences": state.preferences,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "build_seconds": round(build_seconds, 1),
    }
    path = stamp_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_stamp(workspace: Path) -> dict | None:
    """The stamp beside the image, or ``None`` if absent or unreadable."""

    try:
        stamp = json.loads(stamp_path(workspace).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def _installed_version() -> str:
    from jutul_agent import __version__

    return __version__


def clear(workspace: Path) -> bool:
    """Remove the image and its stamp. ``True`` if there was something to remove."""

    directory = sysimage_dir(workspace)
    if not directory.exists():
        return False
    with contextlib.suppress(OSError):
        shutil.rmtree(directory)
    return True


# ---------------------------------------------------------------------------
# The decision.


@dataclass(frozen=True)
class Decision:
    """Whether this session starts from a system image, and what to say if not."""

    status: str
    path: Path | None = None
    # Why the image cannot be used, ready to print. Empty when it can.
    reason: str = ""
    # Things worth mentioning that do not stop the session.
    notes: tuple[str, ...] = ()
    # What to do about it, when rebuilding is not the answer. Empty means it is.
    fix: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.path is not None

    @property
    def blocks(self) -> bool:
        """Whether a workspace asking for a system image has to stop here."""
        return self.status in (MISSING, UNUSABLE, DIVERGENT)

    @property
    def summary(self) -> str:
        """The reason in one line, for reports that have only one to spend.

        The full reason lists every component that moved; this is its headline,
        which is what a status line wants. Whoever needs the detail prints
        ``reason``.
        """
        if not self.reason:
            return ""
        first = self.reason.strip().splitlines()[0]
        return first.strip().rstrip(":")


def decide(workspace: Path, julia_project: Path, *, enabled: bool) -> Decision:
    """Should this session start from the workspace's system image?

    ``enabled`` is the resolved on/off setting. Off is off: an image on disk is
    left alone and nothing is checked, so a workspace that has stopped using one
    pays nothing for still having it.
    """

    if not enabled:
        return Decision(status=OFF)

    image = sysimage_path(workspace)
    if not image.exists():
        return Decision(status=MISSING, reason="no system image has been built for this workspace")

    stamp = read_stamp(workspace)
    if stamp is None:
        return Decision(
            status=UNUSABLE,
            reason=(
                "the system image has no readable stamp, so there is no way to tell "
                "what it was built from"
            ),
        )

    displayless = _missing_display_for(stamp)
    if displayless is not None:
        return displayless

    divergences, notes = _compare(stamp, read_env_state(julia_project))
    if divergences:
        return Decision(status=DIVERGENT, reason=_format_divergences(divergences), notes=notes)
    return Decision(
        status=INCOMPLETE if notes else CURRENT,
        path=image,
        notes=notes,
    )


def _missing_display_for(stamp: dict) -> Decision | None:
    """Refuse an image whose OpenGL binding cannot initialise on this machine.

    A package inside an image runs its ``__init__`` while the process starts,
    before any code of ours. GLFW's opens a window system connection and aborts
    the process when there is none, so baking GLMakie makes a display a condition
    of starting at all, not just of plotting. Loaded the ordinary way it is
    imported only when something plots, which is why this cannot happen without
    an image and why the answer is never to rebuild.
    """

    from jutul_agent.display import plotting_display_available, xvfb_opted_out

    baked = set(stamp.get("versions") or {}) | set(stamp.get("path_packages") or {})
    if not baked.intersection(OPENGL_PACKAGES) or plotting_display_available():
        return None

    if xvfb_opted_out():
        fix = (
            f"Allow a virtual display by unsetting {XVFB_OPT_OUT}:",
            f"    unset {XVFB_OPT_OUT}",
        )
    else:
        fix = ("Give this machine a virtual display by installing Xvfb:", "    xvfb-run")
    return Decision(
        status=UNUSABLE,
        reason=(
            "the image contains GLMakie, whose OpenGL binding initialises as the "
            "process starts, and this machine has no display for it"
        ),
        fix=fix,
    )


def _compare(stamp: dict, state: EnvState) -> tuple[list[tuple[str, list[str]]], tuple[str, ...]]:
    """Diff a stamp against the environment: divergences to block on, notes to print."""

    divergences: list[tuple[str, list[str]]] = []

    if stamp.get("recipe") != RECIPE_VERSION:
        divergences.append(("built by a different version of jutul-agent", []))

    current_julia = julia_version()
    if current_julia and stamp.get("julia") and stamp["julia"] != current_julia:
        divergences.append((f"built with Julia {stamp['julia']}, this is {current_julia}", []))

    if stamp.get("platform") and stamp["platform"] != platform_tag():
        divergences.append(
            (f"built for {stamp['platform']}, this is {platform_tag()}", []),
        )

    stamped_versions = stamp.get("versions") or {}
    changed = [
        f"{name} {stamped_versions[name]} -> {version}"
        for name, version in sorted(state.versions.items())
        if name in stamped_versions and stamped_versions[name] != version
    ]
    if changed:
        divergences.append(("package versions changed", changed))

    stamped_sources = stamp.get("path_packages") or {}
    edited = [
        name
        for name, digest in sorted(state.path_packages.items())
        if name in stamped_sources and stamped_sources[name] != digest
    ]
    if edited:
        divergences.append(("edited since the image was built", edited))

    # Compared even when either side is absent: gaining or losing the file is
    # itself a change of what the baked packages were compiled against.
    if stamp.get("preferences") != state.preferences:
        divergences.append(("LocalPreferences.toml changed since the image was built", []))

    # A package the environment has and the image does not is safe: it loads from
    # its own pkgimage, the way everything did before there was an image at all.
    known = set(stamped_versions) | set(stamped_sources)
    added = sorted((set(state.versions) | set(state.path_packages)) - known)
    notes: tuple[str, ...] = ()
    if added:
        listed = ", ".join(added[:6]) + (f", and {len(added) - 6} more" if len(added) > 6 else "")
        notes = (
            f"the system image does not contain {listed} (installed after it was "
            "built); they load normally, so only startup is a little slower",
        )
    return divergences, notes


def _format_divergences(divergences: list[tuple[str, list[str]]]) -> str:
    lines: list[str] = []
    for headline, items in divergences:
        if items:
            lines.append(f"  {headline}:")
            lines.extend(f"    {item}" for item in items)
        else:
            lines.append(f"  {headline}")
    return "\n".join(lines)


class SysimageUnavailable(RuntimeError):
    """A workspace set to run from a system image cannot use the one it has.

    Carries a message already formatted for a terminal, so a front end prints it
    rather than reformatting a failure it would have to understand first.
    """

    def __init__(self, decision: Decision, *, command: str) -> None:
        super().__init__(refusal(decision, command=command))
        self.decision = decision


# The command a surface tells the user to re-run. A refusal names the way through
# it, and the way through depends on how they got here.
SURFACE_COMMANDS = {"web": "jutul-agent web", "tui": "jutul-agent tui", "cli": "jutul-agent run"}


def refusal(decision: Decision, *, command: str = "jutul-agent web") -> str:
    """The message shown when a workspace wants a system image it cannot use.

    A refusal is a wall, so it carries the way through it: what changed, the one
    command that fixes it, and the flag that starts without it. The component-level
    detail is the point. "Your system image is stale" is an obstacle;
    "geoteric_agentic_demo_julia was edited" is an instruction.
    """

    parts = [
        "This workspace is set to run from a Julia system image, but it cannot be used:",
        "",
        decision.reason if "\n" in decision.reason else f"  {decision.reason}",
        "",
        *(decision.fix or ("Rebuild it with:", "    jutul-agent sysimage build")),
        "",
        "Or start without it, just for this run:",
        f"    {command} --no-sysimage",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# On or off.


def resolve_enabled(explicit: bool | None = None, *, workspace_enabled: bool | None = None) -> bool:
    """Resolve the on/off setting by precedence, highest first.

    ``--sysimage`` / ``--no-sysimage`` (``explicit``) > workspace config >
    ``$JUTUL_AGENT_SYSIMAGE`` > off. The same shape as model resolution, so a
    front end that already knows how one works knows how the other does.

    Off by default because building an image costs tens of minutes: a workspace
    opts in, and after that the flag in its config keeps it opted in.
    """

    if explicit is not None:
        return explicit
    if workspace_enabled is not None:
        return workspace_enabled
    raw = os.environ.get(SYSIMAGE_ENV_VAR)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")
