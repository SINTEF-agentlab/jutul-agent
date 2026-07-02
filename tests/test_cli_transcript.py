"""CLI transcript subcommand against a synthetic trace."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.interfaces.cli import main
from jutul_agent.paths import set_state_home, set_workspace_root, workspace_state_dir


def _session_dir(workspace: Path, state_home: Path, session_id: str) -> Path:
    """Resolve the on-disk session directory for the given workspace.

    Mirrors what the CLI does internally so the test can plant a trace at
    the location the CLI will look it up from.
    """
    set_workspace_root(workspace)
    set_state_home(state_home)
    return workspace_state_dir() / "sessions" / session_id


def _make_trace(workspace: Path, state_home: Path, session_id: str) -> Path:
    from jutul_agent.trace import TraceLog

    sess_dir = _session_dir(workspace, state_home, session_id)
    sess_dir.mkdir(parents=True, exist_ok=True)
    db = sess_dir / "trace.sqlite"
    log = TraceLog(db)
    try:
        log.append("session_start", {"session_id": session_id, "simulator": None})
        log.append("message_user", {"content": "hello"})
        log.append("message_assistant", {"content": "hi there"})
        log.append("session_end", {"session_id": session_id})
    finally:
        log.close()
    return sess_dir


def _cli_flags(workspace: Path, state_home: Path) -> list[str]:
    return ["--workspace", str(workspace), "--state-home", str(state_home)]


def test_transcript_renders_existing_session_to_file(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_home = tmp_path / "state"
    sid = "test-session"
    _make_trace(workspace, state_home, sid)

    code = main(["transcript", sid, *_cli_flags(workspace, state_home)])
    assert code == 0
    out_path = Path(capsys.readouterr().out.strip())
    assert out_path.name == "transcript.html"
    content = out_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "hello" in content
    assert "hi there" in content


def test_transcript_markdown_to_stdout(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_home = tmp_path / "state"
    sid = "test-session-md"
    _make_trace(workspace, state_home, sid)

    code = main(
        [
            "transcript",
            sid,
            "--format",
            "markdown",
            "-o",
            "-",
            *_cli_flags(workspace, state_home),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert f"# Session `{sid}`" in captured.out
    assert "hello" in captured.out
    assert "hi there" in captured.out


def test_transcript_html_to_stdout(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_home = tmp_path / "state"
    sid = "test-session-html"
    _make_trace(workspace, state_home, sid)

    code = main(
        [
            "transcript",
            sid,
            "--format",
            "html",
            "-o",
            "-",
            *_cli_flags(workspace, state_home),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "<!doctype html>" in captured.out
    assert "hello" in captured.out


def test_transcript_bundle_zips_artifacts_from_the_output_dir(tmp_path: Path, capsys) -> None:
    """Plots are written under the workspace output dir, not the state dir; the
    bundle and the embedded images must come from there."""
    import base64
    import zipfile

    from jutul_agent.paths import session_output_dir
    from jutul_agent.trace import TraceLog
    from jutul_agent.trace.schema import ARTIFACT, artifact_payload

    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_home = tmp_path / "state"
    sid = "bundle-session"
    sess_dir = _make_trace(workspace, state_home, sid)
    png = b"\x89PNG\r\n\x1a\n" + b"fake"
    artifacts = session_output_dir(sid) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "plot-test.png").write_bytes(png)
    with TraceLog(sess_dir / "trace.sqlite") as log:
        log.append(
            ARTIFACT,
            artifact_payload(path="artifacts/plot-test.png", mime="image/png", caption="a plot"),
        )

    code = main(["transcript", sid, "--bundle", *_cli_flags(workspace, state_home)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    html_path = Path(next(line for line in lines if line.endswith("transcript.html")))
    encoded = base64.standard_b64encode(png).decode("ascii")
    assert encoded in html_path.read_text(encoding="utf-8")
    bundle = sess_dir / "transcript-bundle.zip"
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        assert "artifacts/plot-test.png" in archive.namelist()


def test_transcript_missing_session_returns_nonzero(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_home = tmp_path / "state"

    code = main(["transcript", "does-not-exist", *_cli_flags(workspace, state_home)])
    captured = capsys.readouterr()
    assert code != 0
    assert "No trace" in captured.err


def test_transcript_uses_last_session_when_no_id(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_home = tmp_path / "state"
    sid = "last-session-test"
    sess_dir = _make_trace(workspace, state_home, sid)
    (sess_dir.parent.parent / "last-session").write_text(sid, encoding="utf-8")

    code = main(["transcript", *_cli_flags(workspace, state_home)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    out_path = Path(captured.out.strip())
    content = out_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content


def test_transcript_no_id_and_no_marker_errors(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_home = tmp_path / "state"

    code = main(["transcript", *_cli_flags(workspace, state_home)])
    captured = capsys.readouterr()
    assert code == 2
    assert "no session id given" in captured.err
