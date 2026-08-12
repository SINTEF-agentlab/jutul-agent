# The Julia system image

A workspace can start its Julia sessions from a **system image**: every package
the environment uses, compiled and linked into a single file that the process
begins with already loaded. It is off by default, costs tens of minutes to build,
and removes most of the wait at the start of every session afterwards.

## What it is for

Precompilation already removes the compiling from a cold start. What it cannot
remove is the loading. Opening, validating and mapping several hundred cache
files takes seconds every time, on a machine where nothing has changed since
yesterday.

A system image is those packages already loaded, serialized once. Measured on a
JutulDarcy workspace with everything precompiled and warm:

| | without an image | from an image |
|---|---|---|
| Kernel ready | 0.8s | 2.8s |
| Simulator and plotting stack loaded | 8.7s | 3.0s |
| Ensemble of two workers ready | 20.5s | 7.5s |

The image is not free at startup: two gigabytes have to be mapped before Julia
runs a line, which is the two seconds the first row costs. It buys back more
than that immediately, because the packages are already there. Distributed
workers inherit the image, so an ensemble saves the most, every worker having
paid the loading separately before.

The trade is build time, disk (an image runs to a few GB), and the staleness
problem below.

## Headless machines are fine, as long as they have xvfb

Headless is the ordinary case on a server and it is unaffected: jutul-agent
starts an Xvfb display for the Julia process, and a session runs from an image
there exactly as it does with a monitor attached, plotting included.

What an image removes is the fallback below that. Loaded the ordinary way,
GLMakie is imported the first time something plots, so a machine with neither a
display nor `xvfb` still runs everything except plotting. Baked into an image,
its OpenGL binding initialises as the process starts, before any code of ours,
and takes the process down when there is no window system to bind to.

So the case to know about is narrow: no display and no `xvfb`, which means either
`JUTUL_AGENT_NO_XVFB` is set or the image was copied somewhere `xvfb` is not
installed. A machine that built its own image already has it, because the build
needs one too. Rather than let Julia abort with a message that explains nothing,
the guard checks this and says which fix applies:

```console
$ jutul-agent run "..."
This workspace is set to run from a Julia system image, but it cannot be used:

  the image contains GLMakie, whose OpenGL binding initialises as the process
  starts, and this machine has no display for it

Allow a virtual display by unsetting JUTUL_AGENT_NO_XVFB:
    unset JUTUL_AGENT_NO_XVFB

Or start without it, just for this run:
    jutul-agent run --no-sysimage
```

## Building one

```console
$ jutul-agent init --sim jutuldarcy --sysimage   # environment, precompile, then image
$ jutul-agent sysimage build                     # rebuild an initialised folder
$ jutul-agent sysimage status                    # what it contains, and whether it still fits
$ jutul-agent sysimage clear                     # remove it, and stop using it
```

`--sysimage` on `init` is remembered in the folder's config, so a later plain
`jutul-agent init` here rebuilds the image without being asked again. A
successful `sysimage build` turns the folder on the same way, and `clear` turns
it off, so the setting and the file on disk cannot drift apart.

Everything in the environment is baked, path-tracked packages included. Those are
usually the ones that matter: the shared `JutulAgent` runtime, each
`JutulAgent<Sim>` warm package, and any Julia package an
[extension](extending-for-your-application.md) contributes. Capabilities are
composed before the build for exactly this reason, so an extended workspace gets
an image of the environment it actually runs.

The image is compiled for the machine that builds it. Pass
`--cpu-target generic` (or another target) to build one that will run on
different hardware, at some cost in speed.

### Windows and the image size limit

Windows refuses to map a DLL whose in-memory span — the PE header's
`SizeOfImage`, not the file's size on disk — is over `0x77000000` (1.86 GiB),
with the unhelpful "%1 is not a valid Win32 application". A full simulator
environment builds to right around that line, so a Windows build makes room in
two ways, and refuses to install an image that is still over it.

It leaves a handful of leaf packages out of the bake (the graph-plot and
table-export paths: CSV, DataFrames, GraphMakie and its layout packages). They
stay in the environment and load the ordinary way at first use, so nothing is
lost but a few seconds, once, on the sessions that use them.

It also sets `[CairoMakie] precompile_workload = false` in the environment's
`LocalPreferences.toml`, because that one workload costs 163 MiB — more than the
limit leaves spare. The poster shapes we actually save are baked by `JutulAgent`'s
own workload instead, so what this costs is a first Cairo render of a plot that
goes off those paths. It is written on Windows only, and only into the generated
environment, so it never follows a checkout elsewhere.

Docstrings are kept: stripping metadata would fit too, but then every `@doc` in
a session answers with nothing, which an agent that reads documentation cannot
afford. Julia 1.13's compressed images should remove the need for any of this.

## Staying in sync

A package inside a system image is never checked against its source. A pkgimage
revalidates on every load; a baked package does not. So editing a package that is
in the image, or resolving a different version of it, does not fail. It runs the
old code, quietly, which on a demo machine is the worst possible outcome.

So a folder set to use an image checks, before every session, that the image
still describes the environment. The check reads the manifest, hashes the sources
of path-tracked packages, and hashes `LocalPreferences.toml`, which is fast
enough to run on every launch and needs no Julia. It has two possible outcomes:

- **Out of date.** A package in the image has changed version, a path-tracked
  package has been edited, or a preference has changed. The session does not
  start. What changed is printed, with the command that fixes it.
- **Incomplete.** The environment has a package the image does not, installed
  after it was built. Nothing is wrong: it loads the ordinary way. This is a
  note, not a refusal.

Preferences are checked because a package that reads one while precompiling has
that value compiled in, exactly as its code is. Changing it afterwards moves
nothing in the manifest, so this is the one kind of staleness that would
otherwise pass unnoticed. The file is compared by what it says rather than how it
is written, so reformatting it is not mistaken for changing a setting.

The refusal is deliberate. A system image that is silently skipped when it stops
matching is the failure this feature exists to prevent, because nobody notices
until the slow path is on screen in front of an audience.

```console
$ jutul-agent web
This workspace is set to run from a Julia system image, but it cannot be used:

  edited since the image was built:
    CapabilityPackage

Rebuild it with:
    jutul-agent sysimage build

Or start without it, just for this run:
    jutul-agent web --no-sysimage
```

`--sysimage` and `--no-sysimage` work on `web`, `tui` and `run`, and override the
folder's setting for one launch. `jutul-agent doctor` reports the image and the
same diagnosis without having to trigger it.

## What a build actually does

1. The environment is prepared exactly as a session would prepare it,
   capabilities included, so the image describes what sessions really load. Any
   warm-up code the capabilities declare runs during the build, so what it
   compiles is baked in and the first real call finds it ready.
2. `PackageCompiler` runs from an environment of its own under the state root,
   pointed at the workspace environment. It never becomes a dependency of the
   workspace, whose manifest is what the image is later checked against.
3. The image is built under a temporary name and **verified before it is
   installed**: the packages really are bound in `Main`, GLMakie renders a
   figure, and Bonito writes its static assets. A system image that fails to
   start would otherwise break every interface at once.
4. Only then is it moved into place, and only then stamped. The stamp is what
   makes an image trusted, so an interrupted or failed build leaves the previous
   image untouched and in use.

Because of step 4, rebuilding is safe to attempt on a machine that is about to be
used. A failure costs time, not the working image.
