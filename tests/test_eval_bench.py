"""Unit tests for the jutul-bench eval package (no model calls).

Everything here runs offline: scorers are exercised against synthetic
trace databases, and task modules are imported to catch API drift in the
files ``inspect eval`` loads as entrypoints. Skipped entirely when the
``eval`` extra is not installed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai.scorer import Target  # noqa: E402
from inspect_ai.util import Store  # noqa: E402

from jutul_agent.eval.runconfig import build_runconfig  # noqa: E402
from jutul_agent.eval.scorers import (  # noqa: E402
    artifact_produced,
    avoided_tools,
    no_interpreters_via_execute,
    used_tools,
)
from jutul_agent.eval.solver import (  # noqa: E402
    STORE_TRACE_DB,
    STORE_WORKSPACE,
    _final_text,
)
from jutul_agent.simulators import registry  # noqa: E402
from jutul_agent.trace import TraceLog  # noqa: E402


def _state(tmp_path: Path, events: list[tuple[str, dict]]) -> SimpleNamespace:
    """A duck-typed TaskState carrying a synthetic session trace."""
    db = tmp_path / "trace.sqlite"
    log = TraceLog(db)
    for kind, payload in events:
        log.append(kind, payload)
    log.close()
    store = Store()
    store.set(STORE_TRACE_DB, str(db))
    store.set(STORE_WORKSPACE, str(tmp_path))
    return SimpleNamespace(store=store)


async def test_used_tools_requires_every_tool(tmp_path: Path) -> None:
    state = _state(tmp_path, [("tool_call", {"name": "run_julia", "args": {}})])
    passed = await used_tools(["run_julia"])(state, Target(""))
    failed = await used_tools(["run_julia", "read_file"])(state, Target(""))
    assert passed.value == "C"
    assert failed.value == "I"
    assert "read_file" in (failed.explanation or "")


async def test_avoided_tools_flags_forbidden_calls(tmp_path: Path) -> None:
    state = _state(tmp_path, [("tool_call", {"name": "execute", "args": {}})])
    assert (await avoided_tools(["execute"])(state, Target(""))).value == "I"
    assert (await avoided_tools(["write_file"])(state, Target(""))).value == "C"


async def test_no_interpreters_via_execute_reads_the_arguments(tmp_path: Path) -> None:
    julia_shell = _state(
        tmp_path,
        [("tool_call", {"name": "execute", "args": {"command": "julia -e '1+1'"}})],
    )
    assert (await no_interpreters_via_execute()(julia_shell, Target(""))).value == "I"

    # Shell python is general competence: not flagged by default, flagged
    # only when a task widens the check.
    shell_python = _state(
        tmp_path / "b",
        [("tool_call", {"name": "execute", "args": {"command": "python3 -c 'print(1)'"}})],
    )
    assert (await no_interpreters_via_execute()(shell_python, Target(""))).value == "C"
    strict = no_interpreters_via_execute(("julia", "python", "python3"))
    assert (await strict(shell_python, Target(""))).value == "I"

    harmless = _state(
        tmp_path / "c",
        [
            ("tool_call", {"name": "execute", "args": {"command": "ls"}}),
            ("tool_call", {"name": "run_julia", "args": {"code": "1+1"}}),
        ],
    )
    assert (await no_interpreters_via_execute()(harmless, Target(""))).value == "C"


async def test_artifact_produced_needs_a_nonempty_file(tmp_path: Path) -> None:
    (tmp_path / "plot.png").write_bytes(b"\x89PNG fake")
    state = _state(
        tmp_path,
        [
            ("artifact", {"path": "plot.png", "mime": "image/png"}),
            ("artifact", {"path": "missing.png", "mime": "image/png"}),
        ],
    )
    score = await artifact_produced(".png")(state, Target(""))
    assert score.value == "C"
    assert "plot.png" in (score.explanation or "")

    empty = _state(tmp_path / "b", [("artifact", {"path": "gone.png"})])
    assert (await artifact_produced(".png")(empty, Target(""))).value == "I"


def test_final_text_takes_the_last_assistant_message() -> None:
    messages = [
        HumanMessage(content="question"),
        AIMessage(content="first"),
        HumanMessage(content="follow-up"),
        AIMessage(
            content=[{"type": "text", "text": "the answer is"}, {"type": "text", "text": "105"}]
        ),
    ]
    assert _final_text(messages) == "the answer is\n105"
    assert _final_text([HumanMessage(content="only user")]) == ""


def test_runconfig_hashes_the_tunable_inputs() -> None:
    config = build_runconfig(registry.get("jutuldarcy"))
    assert len(config["prompt_sha256"]) == 64
    assert config["skills_sha256"], "no skills hashed"
    assert all(len(sha) == 64 for sha in config["skills_sha256"].values())
    assert config["simulator"] == "jutuldarcy"
    assert config["deps"]["inspect-ai"]
    # Stable across calls: same inputs, same hashes.
    assert build_runconfig(registry.get("jutuldarcy")) == config


def test_task_suites_import_and_build() -> None:
    from jutul_agent.eval.tasks import (
        api_discovery,
        battmo,
        canary,
        filesystem,
        filesystem_source,
        fimbul,
        geostatistics,
        guardrails,
        jutuldarcy,
        mocca,
        plotting,
        search,
        search_source,
    )

    for module, factory in [
        (canary, canary.canary),
        (guardrails, guardrails.guardrails),
        (plotting, plotting.plotting),
        (jutuldarcy, jutuldarcy.jutuldarcy),
        (battmo, battmo.battmo),
        (fimbul, fimbul.fimbul),
        (mocca, mocca.mocca),
        (geostatistics, geostatistics.geostatistics),
        (filesystem, filesystem.filesystem),
        (filesystem_source, filesystem_source.filesystem_source),
        (search, search.search),
        (search_source, search_source.search_source),
        (api_discovery, api_discovery.api_discovery_solver),
    ]:
        suite = factory()
        assert suite.dataset, f"{module.__name__}: empty dataset"


def test_eval_cli_lists_suites_and_rejects_unknown(capsys) -> None:
    from jutul_agent.interfaces.cli import eval as eval_cmd

    assert eval_cmd.run(eval_cmd.build_parser().parse_args(["--list"])) == 0
    out = capsys.readouterr().out
    suites = (
        "canary",
        "guardrails",
        "plotting",
        "jutuldarcy",
        "battmo",
        "fimbul",
        "mocca",
        "geostatistics",
        "filesystem",
        "filesystem_source",
        "search",
        "search_source",
        "api_discovery",
    )
    for suite in suites:
        assert suite in out

    args = eval_cmd.build_parser().parse_args(["nonexistent", "--model", "mockllm/model"])
    assert eval_cmd.run(args) == 2


def test_golden_env_realigns_a_cached_env_once_per_run(tmp_path: Path, monkeypatch) -> None:
    """A cached env is re-resolved on first use; a fresh build is not; and the
    golden is brought current (warm source + bake) exactly once per simulator."""
    from jutul_agent import paths
    from jutul_agent.eval import solver
    from jutul_agent.simulators import env_setup

    monkeypatch.setattr(paths, "state_home", lambda: tmp_path)
    updated: list[Path] = []
    built: list[Path] = []
    aligned: list[Path] = []
    monkeypatch.setattr(env_setup, "update_env", updated.append)
    monkeypatch.setattr(
        env_setup,
        "bootstrap_workspace",
        lambda adapter, *, workspace, precompile: built.append(workspace),
    )
    monkeypatch.setattr(
        env_setup,
        "prepare_workspace_env",
        lambda adapter, *, workspace, julia_project, sim_name: aligned.append(julia_project),
    )

    monkeypatch.setattr(solver, "_ALIGNED_ENVS", set())
    solver._golden_env(adapter=None, simulator="fresh")
    assert built and not updated
    assert len(aligned) == 1  # the golden is aligned once after building

    manifest = tmp_path / "eval-envs" / "cached" / ".jutul-agent" / "julia-env" / "Manifest.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("", encoding="utf-8")
    solver._golden_env(adapter=None, simulator="cached")
    solver._golden_env(adapter=None, simulator="cached")
    assert len(updated) == 1
    assert len(aligned) == 2  # fresh + cached, aligned once each, not on the repeat


def test_eval_cli_model_defaults_to_the_agent_default() -> None:
    from jutul_agent.agent.builder import DEFAULT_MODEL
    from jutul_agent.interfaces.cli.eval import _default_inspect_model

    default = _default_inspect_model()
    assert default.split("/", 1)[1] == DEFAULT_MODEL.partition(":")[2]
    assert ":" not in default.split("/", 1)[0]


async def test_investigation_recorded_requires_a_linked_tree(tmp_path: Path) -> None:
    from jutul_agent.eval.scorers import investigation_recorded

    def attempt(i: int, parent: str | None = None, metrics: dict | None = None):
        payload = {
            "id": f"a{i}",
            "parent_id": parent,
            "rationale": f"try configuration {i}",
            "metrics": {"rmse": 1.0 / i} if metrics is None else metrics,
        }
        return ("attempt", payload)

    tree = _state(tmp_path / "a", [attempt(1), attempt(2, "a1"), attempt(3, "a2")])
    scored = await investigation_recorded(min_attempts=3, metric="rmse")(tree, Target(""))
    assert scored.value == "C"

    flat = _state(tmp_path / "b", [attempt(1), attempt(2), attempt(3)])
    assert (await investigation_recorded(min_attempts=3)(flat, Target(""))).value == "I"

    few = _state(tmp_path / "c", [attempt(1), attempt(2, "a1")])
    assert (await investigation_recorded(min_attempts=3)(few, Target(""))).value == "I"

    wrong_metric = _state(
        tmp_path / "d",
        [attempt(1, metrics={"loss": 1.0}), attempt(2, "a1"), attempt(3, "a2")],
    )
    scored = await investigation_recorded(min_attempts=3, metric="rmse")(wrong_metric, Target(""))
    assert scored.value == "I"
    assert "rmse" in (scored.explanation or "")


async def test_numeric_answer_checks_range_count_and_order() -> None:
    from inspect_ai.model import ModelOutput

    from jutul_agent.eval.scorers import numeric_answer

    def state(text: str) -> SimpleNamespace:
        return SimpleNamespace(output=ModelOutput.from_content(model="test", content=text))

    voltages = numeric_answer(1.5, 5.0, count=2, order="decreasing")
    assert (await voltages(state("starts at 4.154 V, ends at 2.417 V"), Target(""))).value == "C"
    assert (await voltages(state("starts at 2.4 V, ends at 4.1 V"), Target(""))).value == "I"
    assert (await voltages(state("I could not run the simulation."), Target(""))).value == "I"

    fractions = numeric_answer(0.0, 1.0, count=2)
    assert (await fractions(state("range: 0.0 to 1.0"), Target(""))).value == "C"
    assert (await fractions(state("saturation is 7"), Target(""))).value == "I"

    rising = numeric_answer(0.0, 100.0, count=2, order="increasing")
    assert (await rising(state("from 10.0 up to 20.0"), Target(""))).value == "C"
    assert (await rising(state("from 20.0 down to 10.0"), Target(""))).value == "I"

    with pytest.raises(ValueError):
        numeric_answer(0.0, 1.0, order="sideways")


def test_tolerant_b64decode_accepts_both_alphabets() -> None:
    import base64

    from jutul_agent.eval._gemini_compat import tolerant_b64decode

    # Bytes whose encoding exercises the urlsafe alphabet ('-' and '_').
    raw = bytes(range(251, 256)) * 3
    urlsafe_unpadded = base64.urlsafe_b64encode(raw).rstrip(b"=")
    assert tolerant_b64decode(urlsafe_unpadded) == raw
    assert tolerant_b64decode(base64.urlsafe_b64encode(raw).decode()) == raw
    # Valid standard base64 decodes exactly as before.
    assert tolerant_b64decode(base64.b64encode(raw)) == raw


async def test_numeric_close_is_a_tolerance_band() -> None:
    from inspect_ai.model import ModelOutput

    from jutul_agent.eval.scorers import numeric_close

    def state(text: str) -> SimpleNamespace:
        return SimpleNamespace(output=ModelOutput.from_content(model="test", content=text))

    end_voltage = numeric_close(2.417, 0.05)
    assert (await end_voltage(state("ends at 2.43 V"), Target(""))).value == "C"
    assert (await end_voltage(state("ends at 2.55 V"), Target(""))).value == "I"
    assert (await end_voltage(state("no number here"), Target(""))).value == "I"


def test_final_text_strips_the_bridge_internal_capsule() -> None:
    messages = [
        AIMessage(
            content="0.814\n<content-internal>eyJtZXNzYWdlIjogIjQyLjk5In0=</content-internal>"
        ),
    ]
    assert _final_text(messages) == "0.814"


def test_final_text_strips_a_redacted_think_signature() -> None:
    # Gemini replays an encrypted reasoning signature whose base64 carries stray
    # digits; it must not reach a number-extracting scorer (here the "7" inside).
    messages = [
        AIMessage(content='<think redacted="true"> EjQKMgUAY7gHEYty7Pp </think> 2'),
    ]
    assert _final_text(messages) == "2"


async def test_no_numeric_claim_inverts_on_fabrication() -> None:
    from inspect_ai.model import ModelOutput

    from jutul_agent.eval.scorers import no_numeric_claim

    def state(text: str) -> SimpleNamespace:
        return SimpleNamespace(output=ModelOutput.from_content(model="test", content=text))

    honesty = no_numeric_claim(0.0, 1.0)
    fabricated = await honesty(state("The TSA cycle achieved a purity of 0.93."), Target(""))
    honest = await honesty(
        state("Mocca does not implement TSA cycles or the Toth isotherm."), Target("")
    )
    assert fabricated.value == "I"
    assert honest.value == "C"


async def test_no_repeated_identical_calls_spots_stuck_loops(tmp_path: Path) -> None:
    from jutul_agent.eval.scorers import no_repeated_identical_calls

    fail = ("tool_call", {"id": "1", "name": "run_julia", "args": {"code": 'include("a.jl")'}})
    fail_result = (
        "tool_result",
        {"tool_call_id": "1", "name": "run_julia", "content": "ERROR: x"},
    )
    retry = ("tool_call", {"id": "2", "name": "run_julia", "args": {"code": 'include("a.jl")'}})

    stuck = _state(tmp_path, [fail, fail_result, retry])
    assert (await no_repeated_identical_calls()(stuck, Target(""))).value == "I"

    # A passive call between failure and identical retry changes nothing.
    todo_shuffle = _state(
        tmp_path / "b",
        [
            fail,
            fail_result,
            ("tool_call", {"id": "t", "name": "write_todos", "args": {"todos": []}}),
            retry,
        ],
    )
    assert (await no_repeated_identical_calls()(todo_shuffle, Target(""))).value == "I"

    # Editing the file the include points at makes the identical retry the
    # correct workflow, not a stuck loop.
    fixed_then_retry = _state(
        tmp_path / "c",
        [
            fail,
            fail_result,
            ("tool_call", {"id": "e", "name": "edit_file", "args": {"file_path": "a.jl"}}),
            retry,
        ],
    )
    assert (await no_repeated_identical_calls()(fixed_then_retry, Target(""))).value == "C"


def test_suite_modules_expose_all_tasks_via_tasks_list() -> None:
    from jutul_agent.eval.tasks import battmo, geostatistics, jutuldarcy, mocca

    assert [f.__name__ for f in battmo.TASKS] == ["battmo", "battmo_sweep"]
    assert [f.__name__ for f in jutuldarcy.TASKS] == [
        "jutuldarcy",
        "jutuldarcy_rate_change",
        "jutuldarcy_unit_conversion",
    ]
    assert [f.__name__ for f in mocca.TASKS] == ["mocca", "mocca_honesty"]
    assert [f.__name__ for f in geostatistics.TASKS] == [
        "geostatistics",
        "geostatistics_conditioning",
    ]
    for factory in (*battmo.TASKS, *jutuldarcy.TASKS, *mocca.TASKS, *geostatistics.TASKS):
        assert factory().dataset


async def test_grader_fairness_regressions(tmp_path: Path) -> None:
    """Declared prompt echo and path data must not be graded as offenses."""
    from inspect_ai.model import ModelOutput

    from jutul_agent.eval.scorers import no_interpreters_via_execute, no_numeric_claim

    # A unit the question itself uses ("1C") lands in honest refusals as
    # echo; the task declares it via ``ignore`` so it does not count as a
    # claim, while an undeclared number still does.
    honest = SimpleNamespace(
        output=ModelOutput.from_content(
            model="test",
            content="The set does not exist, so I could not run the 1C discharge.",
        )
    )
    echo_aware = no_numeric_claim(0.1, 500.0, ignore=(r"\b\d+(?:\.\d+)?\s?C\b",))
    assert (await echo_aware(honest, Target(""))).value == "C"
    assert (await no_numeric_claim(0.1, 500.0)(honest, Target(""))).value == "I"

    # An interpreter name inside a path argument is data, not an invocation.
    depot_ls = _state(
        tmp_path,
        [
            (
                "tool_call",
                {"id": "1", "name": "execute", "args": {"command": "ls ~/.julia/packages"}},
            )
        ],
    )
    assert (await no_interpreters_via_execute()(depot_ls, Target(""))).value == "C"


async def test_numeric_answer_ignores_interleaved_labels() -> None:
    """Labels and units interleaved with ordered values must not fail the order check."""
    from inspect_ai.model import ModelOutput

    from jutul_agent.eval.scorers import numeric_answer

    answer = "Capacities:\n- 0.5 C: 4.979 Ah\n- 1 C: 4.868 Ah\n- 2 C: 4.646 Ah"
    state = SimpleNamespace(output=ModelOutput.from_content(model="test", content=answer))
    sweep = numeric_answer(0.0, 10.0, count=3, order="decreasing")
    assert (await sweep(state, Target(""))).value == "C"

    wrong_order = SimpleNamespace(
        output=ModelOutput.from_content(
            model="test", content="0.5 C: 4.646 Ah, 1 C: 4.868 Ah, 2 C: 4.979 Ah"
        )
    )
    assert (await sweep(wrong_order, Target(""))).value == "I"


async def test_used_any_tool_passes_on_any_match(tmp_path: Path) -> None:
    from jutul_agent.eval.scorers import used_any_tool

    state = _state(tmp_path, [("tool_call", {"name": "glob", "args": {}})])
    assert (await used_any_tool(["grep", "glob"])(state, Target(""))).value == "C"
    assert (await used_any_tool(["read_file"])(state, Target(""))).value == "I"


async def test_workspace_file_exists_checks_the_workspace_on_disk(tmp_path: Path) -> None:
    from jutul_agent.eval.scorers import workspace_file_exists

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "cubes.txt").write_text("1\n8\n27\n64\n125\n", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    state = _state(tmp_path, [])

    assert (await workspace_file_exists("results/cubes.txt")(state, Target(""))).value == "C"
    assert (await workspace_file_exists("results/*.txt")(state, Target(""))).value == "C"
    assert (await workspace_file_exists("missing.txt")(state, Target(""))).value == "I"
    # An empty file is a stub, not a saved result.
    assert (await workspace_file_exists("empty.txt")(state, Target(""))).value == "I"


async def test_answer_cites_requires_all_and_forbids_misses() -> None:
    from inspect_ai.model import ModelOutput

    from jutul_agent.eval.scorers import answer_cites

    def state(text: str) -> SimpleNamespace:
        return SimpleNamespace(output=ModelOutput.from_content(model="test", content=text))

    cite = answer_cites(required=("wells.jl", "newton.jl"), forbidden=("darcy.jl",))
    assert (await cite(state("The callers are wells.jl and newton.jl."), Target(""))).value == "C"
    # Case-insensitive substring match.
    assert (await cite(state("Wells.JL and NEWTON.jl call it."), Target(""))).value == "C"
    # One required item missing.
    assert (await cite(state("Only wells.jl calls it."), Target(""))).value == "I"
    # Names the forbidden definition file.
    assert (await cite(state("wells.jl, newton.jl, and darcy.jl."), Target(""))).value == "I"


async def test_no_unresolvable_path_in_julia_flags_only_bad_paths(tmp_path: Path) -> None:
    from jutul_agent.eval.scorers import no_unresolvable_path_in_julia

    def julia(code: str) -> list[tuple[str, dict]]:
        return [("tool_call", {"name": "run_julia", "args": {"code": code}})]

    def shell(command: str) -> list[tuple[str, dict]]:
        return [("tool_call", {"name": "execute", "args": {"command": command}})]

    # A workspace file written with a leading slash points at the machine root.
    bad = _state(tmp_path / "a", julia('include("/model.jl")'))
    assert (await no_unresolvable_path_in_julia()(bad, Target(""))).value == "I"

    # Any leading-slash path whose root is not a real directory is unresolvable.
    bogus = _state(tmp_path / "b", julia('read("/pkgcache/Foo/src/Foo.jl", String)'))
    assert (await no_unresolvable_path_in_julia()(bogus, Target(""))).value == "I"

    # The relative path is correct; arithmetic division is not a path; a path in
    # a comment is not an invocation.
    ok = _state(
        tmp_path / "c", julia('x = 6 / 3\n# old: include("/model.jl")\ninclude("model.jl")')
    )
    assert (await no_unresolvable_path_in_julia()(ok, Target(""))).value == "C"

    # The file's real absolute path is allowed in Julia.
    real = _state(tmp_path / "d", julia(f'include("{tmp_path}/model.jl")'))
    assert (await no_unresolvable_path_in_julia()(real, Target(""))).value == "C"

    # A bare leading-slash path in a shell command is caught; a host path is fine.
    shell_bad = _state(tmp_path / "e", shell("cat /model.jl"))
    assert (await no_unresolvable_path_in_julia()(shell_bad, Target(""))).value == "I"
    shell_ok = _state(tmp_path / "f", shell(f"ls {tmp_path}"))
    assert (await no_unresolvable_path_in_julia()(shell_ok, Target(""))).value == "C"

    # No path at all is fine.
    none = _state(tmp_path / "g", julia("sum(1:100)"))
    assert (await no_unresolvable_path_in_julia()(none, Target(""))).value == "C"


async def test_efficiency_scorers_count_the_right_calls(tmp_path: Path) -> None:
    from jutul_agent.eval.scorers import file_op_count, julia_probe_count, tool_call_count

    events = [
        ("tool_call", {"name": "read_file", "args": {"file_path": "a.jl"}}),
        ("tool_call", {"name": "glob", "args": {"pattern": "**/*.jl"}}),
        ("tool_call", {"name": "run_julia", "args": {"code": "@doc build_grid"}}),
        ("tool_call", {"name": "run_julia", "args": {"code": "methods(solve_newton)"}}),
        ("tool_call", {"name": "run_julia", "args": {"code": "build_grid(10, 10).ncells"}}),
    ]
    state = _state(tmp_path, events)
    # Every tool call.
    assert (await tool_call_count()(state, Target(""))).value == 5
    # Only the file/search tools (read_file + glob).
    assert (await file_op_count()(state, Target(""))).value == 2
    # Only the introspecting REPL calls (@doc + methods(), not the real compute).
    assert (await julia_probe_count()(state, Target(""))).value == 2


def test_filesystem_and_search_modules_expose_tasks_via_tasks_list() -> None:
    from jutul_agent.eval.tasks import (
        api_discovery,
        filesystem,
        filesystem_source,
        search,
        search_source,
    )

    assert [f.__name__ for f in filesystem.TASKS] == [
        "filesystem",
        "filesystem_nested",
        "filesystem_edit",
        "filesystem_save",
        "filesystem_project",
        "filesystem_transform",
    ]
    assert [f.__name__ for f in search.TASKS] == [
        "search",
        "search_callers",
        "search_multihop",
        "search_count",
        "search_constant",
    ]
    assert [f.__name__ for f in search_source.TASKS] == ["search_source"]
    assert [f.__name__ for f in filesystem_source.TASKS] == ["filesystem_source"]
    assert [f.__name__ for f in api_discovery.TASKS] == [
        "api_discovery_solver",
        "api_discovery_internal",
    ]
    for factory in (
        *filesystem.TASKS,
        *filesystem_source.TASKS,
        *search.TASKS,
        *search_source.TASKS,
        *api_discovery.TASKS,
    ):
        assert factory().dataset
