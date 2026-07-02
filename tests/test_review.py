"""Tests for the session reviewer (no real model calls — the model is mocked)."""

from __future__ import annotations

import sys
import types

import pytest

from jutul_agent.review.findings import (
    Finding,
    ReviewReport,
    append_report,
    load_reports,
    now_iso,
)
from jutul_agent.review.reviewer import _coerce_text, _extract_json, review_transcript
from jutul_agent.review.settings import review_enabled, review_model


@pytest.fixture(autouse=True)
def _state_home(tmp_path, monkeypatch):
    from jutul_agent import paths

    monkeypatch.setattr(paths, "_state_home_override", tmp_path)
    return tmp_path


def test_findings_store_round_trip():
    report = ReviewReport(
        session_id="s1",
        title="t",
        model="m",
        created_at="2026-01-01T00:00:00+00:00",
        summary="ok-ish",
        findings=[
            Finding(
                "validation-gap",
                "high",
                "Permeability in mD",
                "perm=500",
                "convert",
                "case-validation",
            )
        ],
    )
    append_report(report)
    append_report(ReviewReport("s2", "", "m", "2026", "clean", []))
    loaded = load_reports()
    assert [r.session_id for r in loaded] == ["s1", "s2"]
    assert loaded[0].findings[0].category == "validation-gap"
    assert loaded[1].ok is True


def test_load_reports_skips_malformed_lines(_state_home):
    from jutul_agent.review.findings import review_log_path

    path = review_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"session_id": "good", "findings": []}\nnot json\n', encoding="utf-8")
    assert [r.session_id for r in load_reports()] == ["good"]


def test_finding_from_dict_tolerates_missing_keys():
    f = Finding.from_dict({"title": "x"})
    assert f.title == "x"
    assert f.category == "other"
    assert f.severity == "medium"
    assert f.fix_target == "other"


def test_extract_json_handles_fences_and_prose():
    bare = '{"summary": "a", "findings": []}'
    assert _extract_json(bare)["summary"] == "a"
    fenced = "Here you go:\n```json\n" + bare + "\n```\nthanks"
    assert _extract_json(fenced)["summary"] == "a"
    prose = "Sure. " + bare + " Done."
    assert _extract_json(prose)["findings"] == []


def test_extract_json_raises_without_object():
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_coerce_text_flattens_parts():
    assert _coerce_text("plain") == "plain"
    assert _coerce_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"


def test_settings_env(monkeypatch):
    monkeypatch.delenv("JUTUL_AGENT_REVIEW", raising=False)
    monkeypatch.delenv("JUTUL_AGENT_REVIEW_MODEL", raising=False)
    assert review_enabled() is False
    assert review_model() == "openai:gpt-5.4"
    monkeypatch.setenv("JUTUL_AGENT_REVIEW", "1")
    monkeypatch.setenv("JUTUL_AGENT_REVIEW_MODEL", "openai:gpt-5.4-mini")
    assert review_enabled() is True
    assert review_model() == "openai:gpt-5.4-mini"


class _FakeModel:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, _messages):
        return types.SimpleNamespace(content=self._content)


async def test_review_transcript_parses_findings(monkeypatch):
    reply = (
        '```json\n{"summary": "one unit error", "findings": ['
        '{"category": "validation-gap", "severity": "high", "title": "mD permeability", '
        '"evidence": "perm=500", "suggestion": "convert_to_si", "fix_target": "case-validation"}'
        "]}\n```"
    )
    fake = types.ModuleType("langchain.chat_models")
    fake.init_chat_model = lambda _id: _FakeModel(reply)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake)

    report = await review_transcript(
        "agent set permeability=500", session_id="sX", title="bad", model_id="openai:gpt-5.4-mini"
    )
    assert report.session_id == "sX"
    assert len(report.findings) == 1
    assert report.findings[0].fix_target == "case-validation"
    assert report.summary == "one unit error"


# ---- curated issue store + curation ----------------------------------------


def _report(session_id, *findings, created="2026-06-17T10:00:00+00:00"):
    return ReviewReport(session_id, "t", "m", created, "sum", list(findings))


def test_issue_store_and_merge_bookkeeping():
    from jutul_agent.review.issues import load_issues, merge_finding, new_issue, save_issues

    rep_a = _report("sess-A")
    f1 = Finding("validation-gap", "medium", "Perm in mD", "perm=500", "convert", "case-validation")
    issue = new_issue(f1, rep_a, title=None, existing={})
    save_issues({issue.id: issue})

    issues = load_issues()
    assert issues[issue.id].count == 1
    # A higher-severity recurrence in a new session bumps severity and tracks both.
    rep_b = _report("sess-B")
    f2 = Finding(
        "validation-gap", "high", "Perm too large", "perm=250", "convert", "case-validation"
    )
    merge_finding(issues[issue.id], f2, rep_b)
    assert issues[issue.id].count == 2
    assert issues[issue.id].severity == "high"
    assert issues[issue.id].sessions == ["sess-A", "sess-B"]
    assert len(issues[issue.id].examples) == 2


def test_unique_id_avoids_collisions():
    from jutul_agent.review.issues import unique_id

    existing = {"perm-in-md": object(), "perm-in-md-2": object()}
    assert unique_id("Perm in mD", existing) == "perm-in-md-3"


def test_set_status_round_trips():
    from jutul_agent.review.issues import load_issues, new_issue, save_issues, set_status

    issue = new_issue(
        Finding("agent-error", "low", "x", "ev", "s", "skill"),
        _report("s"),
        title=None,
        existing={},
    )
    save_issues({issue.id: issue})
    assert set_status(issue.id, "fixed") is True
    assert load_issues()[issue.id].status == "fixed"
    assert set_status("nope", "fixed") is False


def test_delete_issue_removes_it():
    from jutul_agent.review.issues import delete_issue, load_issues, new_issue, save_issues

    issue = new_issue(
        Finding("agent-error", "low", "x", "ev", "s", "skill"),
        _report("s"),
        title=None,
        existing={},
    )
    save_issues({issue.id: issue})
    assert delete_issue(issue.id) is True
    assert load_issues() == {}
    assert delete_issue("nope") is False


def test_is_stale_only_when_version_diverges():
    from jutul_agent.review.issues import Issue, is_stale

    base = dict(
        id="i",
        title="t",
        category="agent-error",
        fix_target="skill",
        severity="low",
        count=1,
        first_seen="2026",
        last_seen="2026",
    )
    open_old = Issue(status="open", last_version="0.1.dev100", **base)
    assert is_stale(open_old, "0.1.dev112") is True
    # Same version, no version, or non-open status -> not stale.
    assert is_stale(Issue(status="open", last_version="0.1.dev112", **base), "0.1.dev112") is False
    assert is_stale(Issue(status="open", last_version="", **base), "0.1.dev112") is False
    assert is_stale(Issue(status="fixed", last_version="0.1.dev100", **base), "0.1.dev112") is False


def test_fix_prompt_includes_evidence_and_resolve_command():
    from jutul_agent.review.issues import new_issue
    from jutul_agent.review.prompt import fix_prompt

    f = Finding("validation-gap", "high", "Perm in mD", "perm=500 left in mD", "convert", "skill")
    issue = new_issue(f, _report("sess-A"), title=None, existing={})
    text = fix_prompt(issue, transcript_paths=["/tmp/sess-A.html"])
    assert "perm=500 left in mD" in text  # evidence carried through
    assert f"jutul-agent review resolve {issue.id}" in text  # closes the loop
    assert "/tmp/sess-A.html" in text  # transcript pointer


async def test_curate_creates_then_merges(monkeypatch):
    from jutul_agent.review.curate import curate_report
    from jutul_agent.review.issues import load_issues

    # The matcher only calls the (mocked) model when a provider key is present.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # First report (no existing issues) -> all created, no model call needed.
    f_perm = Finding(
        "validation-gap", "high", "Perm in mD", "perm=500", "convert", "case-validation"
    )
    f_dt = Finding("agent-error", "medium", "dt in days", "dt=30", "use day", "skill")
    await curate_report(_report("sess-A", f_perm, f_dt), model_id="openai:gpt-5.4-mini")
    issues = load_issues()
    assert len(issues) == 2
    perm_id = next(i.id for i in issues.values() if i.fix_target == "case-validation")

    # Second report: the matcher (mocked) maps a reworded perm finding to the same issue.
    fake = types.ModuleType("langchain.chat_models")
    decision = f'{{"decisions": [{{"finding": 0, "issue": "{perm_id}"}}]}}'
    fake.init_chat_model = lambda _id: _FakeModel(decision)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake)

    f_perm2 = Finding(
        "validation-gap", "high", "Perm not SI", "perm=250", "convert", "case-validation"
    )
    await curate_report(_report("sess-B", f_perm2), model_id="openai:gpt-5.4-mini")
    issues = load_issues()
    assert len(issues) == 2  # merged, not a new issue
    assert issues[perm_id].count == 2
    assert issues[perm_id].sessions == ["sess-A", "sess-B"]


async def test_ingest_findings_logs_without_a_model_call():
    """The coding-agent path: ingest external findings with curate=False -> no API."""
    from jutul_agent.review import ingest_findings, load_reports

    data = {
        "summary": "one unit error",
        "findings": [
            {
                "category": "validation-gap",
                "severity": "high",
                "title": "Perm in mD",
                "evidence": "permeability = 200.0",
                "suggestion": "use 200*milli*Darcy",
                "fix_target": "case-validation",
            }
        ],
    }
    report = await ingest_findings(
        data, session_id="sess-X", model_id="openai:gpt-5.4", source="coding-agent", curate=False
    )
    assert report.model == "coding-agent"
    assert report.findings[0].fix_target == "case-validation"
    assert load_reports()[-1].session_id == "sess-X"


async def test_curate_falls_back_to_titles_when_no_api_key(monkeypatch):
    """Offline path: with no provider key, curation merges by exact title, no model call."""
    from jutul_agent.review import curate
    from jutul_agent.review.curate import curate_report
    from jutul_agent.review.issues import load_issues

    # No reviewer-model credential available -> deterministic matcher only.
    monkeypatch.setattr(curate, "missing_credential", lambda _model: "OPENAI_API_KEY")

    def _boom(_id):  # a model call here would be a bug
        raise AssertionError("must not call the model when the key is missing")

    fake = types.ModuleType("langchain.chat_models")
    fake.init_chat_model = _boom
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake)

    f = Finding("tooling-gap", "high", "CaseValidation missing", "UndefVarError", "pin", "code")
    await curate_report(_report("sess-A", f), model_id="openai:gpt-5.4")
    # A same-titled finding from another session folds into the one issue.
    await curate_report(_report("sess-B", f), model_id="openai:gpt-5.4")
    issues = load_issues()
    assert len(issues) == 1
    only = next(iter(issues.values()))
    assert only.count == 2
    assert only.sessions == ["sess-A", "sess-B"]


# ---- cross-workspace discovery ---------------------------------------------


def _make_session(state_home, ws, sid, *, title=None):
    """Create a minimal on-disk session (a trace file is enough to discover it)."""
    d = state_home / "workspaces" / ws / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "trace.sqlite").write_text("", encoding="utf-8")
    if title is not None:
        (d / "title").write_text(title, encoding="utf-8")
    return d


def test_discover_sessions_spans_workspaces_and_flags_reviewed(_state_home):
    from jutul_agent.review.discovery import discover_sessions

    _make_session(_state_home, "ws1", "2026-06-17-1000-aaaa", title="first")
    _make_session(_state_home, "ws2", "2026-06-17-1200-bbbb", title="second")
    # A logged review marks one session as reviewed.
    append_report(ReviewReport("2026-06-17-1000-aaaa", "first", "m", "2026", "ok", []))

    found = discover_sessions()
    ids = [s.session_id for s in found]
    assert ids == ["2026-06-17-1200-bbbb", "2026-06-17-1000-aaaa"]  # newest first
    assert {s.workspace for s in found} == {"ws1", "ws2"}
    assert next(s for s in found if s.session_id.endswith("aaaa")).reviewed is True

    pending = discover_sessions(pending_only=True)
    assert [s.session_id for s in pending] == ["2026-06-17-1200-bbbb"]


def test_find_session_by_prefix(_state_home):
    from jutul_agent.review.discovery import find_session

    _make_session(_state_home, "ws1", "2026-06-17-1000-aaaa")
    assert find_session("2026-06-17-1000-aaaa").workspace == "ws1"
    assert find_session("2026-06-17-1000").session_id.endswith("aaaa")  # unique prefix
    assert find_session("nope") is None


def test_session_simulator_and_ground_truth_read_from_trace(tmp_path):
    from jutul_agent.review.discovery import session_ground_truth, session_simulator
    from jutul_agent.trace import TraceLog

    trace = tmp_path / "trace.sqlite"
    log = TraceLog(trace)
    log.append("session_start", {"session_id": "s", "simulator": "battmo"})
    log.append("eval_target", {"expected": "about 203.7 bar"})
    assert session_simulator(trace) == "battmo"
    assert session_ground_truth(trace) == "about 203.7 bar"
    assert session_simulator(tmp_path / "missing.sqlite") is None
    # A session with no eval target has no ground truth.
    plain = tmp_path / "plain.sqlite"
    TraceLog(plain).append("session_start", {"simulator": "jutuldarcy"})
    assert session_ground_truth(plain) is None


def test_build_user_message_includes_ground_truth():
    from jutul_agent.review.prompt import build_user_message

    msg = build_user_message("trace", simulator="jutuldarcy", ground_truth="about 203.7 bar")
    assert "EVAL run" in msg and "203.7 bar" in msg
    assert "EVAL run" not in build_user_message("trace", simulator=None)


def test_eval_review_context_combines_expected_and_verdict(tmp_path):
    from jutul_agent.review.discovery import eval_review_context, session_eval_result
    from jutul_agent.trace import TraceLog

    trace = tmp_path / "trace.sqlite"
    log = TraceLog(trace)
    log.append("eval_target", {"expected": "about 203.7 bar"})
    log.append("eval_result", {"passed": False, "task": "jutuldarcy_unit_conversion"})
    assert session_eval_result(trace)["passed"] is False
    ctx = eval_review_context(trace)
    assert "expected about 203.7 bar" in ctx and "FAIL" in ctx


def test_link_eval_results_writes_verdict_onto_session(_state_home):
    import types

    pytest.importorskip("inspect_ai")  # _passed grades via inspect_ai (the eval extra)

    from jutul_agent.review.discovery import session_eval_result
    from jutul_agent.review.eval_link import link_eval_results
    from jutul_agent.trace import TraceLog

    d = _make_session(_state_home, "eval-jutuldarcy", "2026-06-18-0900-eval")
    # Give the session a real trace so find_session/append work.
    TraceLog(d / "trace.sqlite").append("session_start", {"simulator": "jutuldarcy"})

    # A minimal stand-in for an inspect EvalLog with one scored sample.
    sample = types.SimpleNamespace(
        store={"jutul/session_id": "2026-06-18-0900-eval"},
        scores={"numeric_close": types.SimpleNamespace(value="C")},
    )
    log = types.SimpleNamespace(
        samples=[sample], eval=types.SimpleNamespace(task="jutuldarcy_unit_conversion")
    )
    assert link_eval_results([log]) == 1
    result = session_eval_result(d / "trace.sqlite")
    assert result["passed"] is True and result["task"] == "jutuldarcy_unit_conversion"


def test_passed_ignores_non_grade_metric_scorers():
    """Metric scorers (counts, not CORRECT/INCORRECT) don't gate the verdict."""
    import types

    pytest.importorskip("inspect_ai")  # _passed grades via inspect_ai (the eval extra)

    from jutul_agent.review.eval_link import _passed

    def g(v):
        return types.SimpleNamespace(value=v)

    assert _passed({"answer": g("C"), "tool_call_count": g("11")}) is True
    assert _passed({"answer": g("I"), "tool_call_count": g("11")}) is False
    assert _passed({"tool_call_count": g("3")}) is False  # nothing graded


def test_critic_prompt_is_general_and_injects_only_active_simulator_hints():
    from jutul_agent.review.prompt import SYSTEM, full_prompt

    low = SYSTEM.lower()
    assert "permeability" not in low and "casevalidation" not in low and "voltage" not in low
    jd = full_prompt("t", simulator="jutuldarcy").lower()
    assert "permeability" in jd and "purities" not in jd
    mo = full_prompt("t", simulator="mocca").lower()
    assert "purities" in mo and "permeability" not in mo
    assert "permeability" not in full_prompt("t", simulator=None).lower()


def test_eval_sessions_state_root_is_discoverable(_state_home):
    from jutul_agent.paths import eval_sessions_state_root
    from jutul_agent.review.discovery import discover_sessions

    root = eval_sessions_state_root("jutuldarcy")
    assert root == _state_home / "workspaces" / "eval-jutuldarcy"
    # A session written under that root is found by discovery like any other.
    d = root / "sessions" / "2026-06-17-0900-eval"
    d.mkdir(parents=True)
    (d / "trace.sqlite").write_text("", encoding="utf-8")
    assert any(s.workspace == "eval-jutuldarcy" for s in discover_sessions())


def test_server_apply_action(_state_home):
    from jutul_agent.review.issues import load_issues, new_issue, save_issues
    from jutul_agent.review.server import _apply_action

    issue = new_issue(
        Finding("agent-error", "low", "x", "ev", "s", "skill"),
        _report("s"),
        title=None,
        existing={},
    )
    save_issues({issue.id: issue})
    assert _apply_action("dismiss", issue.id) == (True, "dismissed")
    assert load_issues()[issue.id].status == "dismissed"
    assert _apply_action("reopen", issue.id) == (True, "open")
    assert _apply_action("delete", issue.id)[0] is True
    assert load_issues() == {}
    assert _apply_action("bogus", "x")[0] is False


# ---- dashboard + export -----------------------------------------------------


def _seed_issue_and_report(state_home, sid, finding, *, simulator=""):
    """One discoverable session with a logged review and a matching curated issue."""
    from jutul_agent.review.issues import new_issue, save_issues

    _make_session(state_home, "ws1", sid, title="perm run")
    report = ReviewReport(
        sid, "perm run", "coding-agent", now_iso(), "s", [finding], simulator=simulator
    )
    append_report(report)
    issue = new_issue(finding, report, title=None, existing={})
    save_issues({issue.id: issue})
    return issue


def test_build_data_ranks_by_priority_and_tags_simulator(_state_home):
    from jutul_agent.review.dashboard import build_data

    f = Finding("validation-gap", "high", "Perm in mD", "perm=500", "convert", "case-validation")
    _seed_issue_and_report(_state_home, "2026-06-17-1000-aaaa", f, simulator="jutuldarcy")

    data = build_data()
    row = data["issues"][0]
    assert row["title"] == "Perm in mD"
    assert row["priority"] > 0
    assert data["stats"]["by_simulator"] == [{"name": "jutuldarcy", "count": 1}]
    assert row["last_session"][:10] == "2026-06-17"  # from the session, not the review


def test_render_page_embeds_data_and_transcripts():
    from jutul_agent.review.dashboard import render_page

    data = {"stats": {}, "issues": [], "reviews": []}
    html = render_page(data, transcripts={"sid": "<html>stub</html>"})
    assert '"stats"' in html and "stub" in html


def test_export_html_and_markdown(_state_home, monkeypatch):
    from jutul_agent.review import export

    monkeypatch.setattr(export, "render_trace_html", lambda _p: "<html>embedded transcript</html>")
    f = Finding("validation-gap", "high", "Perm in mD", "perm=500", "convert", "case-validation")
    _seed_issue_and_report(_state_home, "2026-06-17-1000-aaaa", f)

    html = export.export_html(_state_home / "out.html")
    assert "embedded transcript" in html.read_text(encoding="utf-8")  # transcript inlined

    md = export.export_markdown(_state_home / "out.md")
    text = md.read_text(encoding="utf-8")
    assert "Perm in mD" in text and "perm=500" in text
